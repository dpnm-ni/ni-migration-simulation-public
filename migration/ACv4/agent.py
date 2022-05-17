import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from util.config import NUM_EDGE_DC

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 32
# learning_rate: 0.001~2 (DDPG 논문), 0.01 (도영 DQN), 0.0001 (Cartpole)
LEARNING_RATE = 0.001
THRESHOLD_GRAD_NORM = 1
GAMMA = 0.98

# https://doheejin.github.io/pytorch/2021/09/22/pytorch-autograd-detect-anomaly.html
# torch.autograd.set_detect_anomaly(True)


class ActorCriticv4MigrationAgent:
    def __init__(self, edgeDC_id, dim_mig_nn_input):
        self.edgeDC_id = edgeDC_id
        self.net = Net(dim_mig_nn_input)

        self.num_epi = 0
        self.data = []

        self.migration_algorithm = None

    def attach_algorithm(self, migration_algorithm):
        self.migration_algorithm = migration_algorithm

    def put_data(self, item):
        self.data.append(item)

    def train(self):
        if len(self.data) == 0:
            return

        loss_lst = []
        for transition in self.data:
            s, a, r, s_prime = transition

            td_target = r + GAMMA * self.net.v(s_prime)
            delta = td_target - self.net.v(s)
            # Note: ensure torch.sum(pi[i]) == 1
            pi = self.net.pi(s, softmax_dim=1)
            # pi is 3-dim, so convert a into the same dim as well just to fetch target indices.
            pi_a = pi.gather(1, a.view(-1, 1, 1))
            pi_utilization_func = -torch.log(pi_a) * delta.detach()
            v_loss_func = F.smooth_l1_loss(self.net.v(s), td_target.detach())
            loss = pi_utilization_func + v_loss_func

            loss_lst.append(loss.mean())

            # FIXME: version 1
            # self.net.optimizer.zero_grad()
            # loss.mean().backward()
            # # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=THRESHOLD_GRAD_NORM)
            # self.net.optimizer.step()

        # FIXME: version2. loss_lst에 들어있는 n개 loss.mean의 평균 -> 최종 loss
        self.net.optimizer.zero_grad()
        torch.stack(loss_lst).mean().backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=THRESHOLD_GRAD_NORM)
        self.net.optimizer.step()

        self.data = []

    def get_action(self, sub_state):
        # Note: ensure torch.sum(fitness_scores[i]) == 1
        fitness_scores = self.net.pi(sub_state)
        probs = fitness_scores.detach().transpose(0, 1)
        dest_edge_id = Categorical(probs=probs).sample().item()

        return dest_edge_id


class Net(nn.Module):
    def __init__(self, dim_mig_nn_input):
        super(Net, self).__init__()

        # Common
        self.input = nn.Linear(dim_mig_nn_input, NUM_NEURONS)
        self.hidden1 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.hidden2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)

        # Actor
        # self.fc_pi = nn.Linear(NUM_NEURONS, NUM_EDGE_DC + 1)
        self.fc_pi = nn.Linear(NUM_NEURONS, 1)

        # Critic
        self.fc_v = nn.Linear(NUM_NEURONS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        prob = F.softmax(self.fc_pi(x), dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        v = self.fc_v(x)
        return v
