import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util.config import NUM_EDGE_DC

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 32
# learning_rate: 0.001~2 (DDPG 논문), 0.01 (도영 DQN), 0.0001 (Cartpole)
LEARNING_RATE = 0.001
THRESHOLD_GRAD_NORM = 1
# replay mem capacity: 10000 (Cartpole), 2500 (도영 DQN)
# FIXME: 1 episode에 대략 몇 개 저장되는지 확인 후 적절한 값 설정
BUFFER_CAPACITY = 50000
# FIXME: 1 에피소드에 각 엣지별 평균 50~100개 transition 생성됨. 에피소드 #5부터 학습 시작하도록
LEAST_SIZE_TO_LEARN = 1000
# mini-batch sampling size: 32 (Cartpole), 16 (도영)
BATCH_SIZE = 10
NUM_ROLLOUT = 15
GAMMA = 0.98

# https://doheejin.github.io/pytorch/2021/09/22/pytorch-autograd-detect-anomaly.html
# torch.autograd.set_detect_anomaly(True)


class DQNv3MigrationAgent:
    def __init__(self, edgeDC_id, dim_mig_nn_input):
        self.edgeDC_id = edgeDC_id

        self.q_net = Net(dim_mig_nn_input)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.target_q_net = Net(dim_mig_nn_input)
        self.update_target_q_function()

        self.num_epi = 0
        self.memory = ReplayBuffer()

        self.migration_algorithm = None

    def attach_algorithm(self, migration_algorithm):
        self.migration_algorithm = migration_algorithm

    def put_data(self, item):
        self.memory.put(item)

    def train(self):
        if self.memory.size() > LEAST_SIZE_TO_LEARN:
            # FIXME: 아래 mean 연산 때문에 가능한 batch size (divisor) 적게하고 rollout을 키우는 방향으로 수정할 것.
            #  ACv3에서 에피소드당 total_sim_time/(mig_interval*num_rollout) = 1680/10*10 = 17회 정도 수행하므로 비슷하게.
            for i in range(NUM_ROLLOUT):
                loss_lst = []
                mini_batch = random.sample(self.memory.buffer, BATCH_SIZE)
                for transition in mini_batch:
                    s, a, r, s_prime = transition

                    # max a s.t max Q(s, a)
                    q_a = self.q_net(s).gather(1, a.view(-1, 1, 1))
                    max_q_prime = self.target_q_net(s_prime).max(1)[0].unsqueeze(1)
                    target = r + GAMMA * max_q_prime
                    loss = F.smooth_l1_loss(q_a, target)

                    loss_lst.append(loss)

                    # FIXME: version 1. batch size * rollout 만큼 backward 발생하여 too slow.
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=THRESHOLD_GRAD_NORM)
                    # self.optimizer.step()

                # FIXME: version2. loss_lst에 들어있는 n개 loss의 평균/중위값 -> 최종 loss
                self.optimizer.zero_grad()
                torch.stack(loss_lst).mean().backward()
                # torch.stack(loss_lst).median().backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=THRESHOLD_GRAD_NORM)
                self.optimizer.step()

    def update_target_q_function(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def get_action(self, sub_state):
        # Linear annealing from 10% to 1% (when num_epi is 100).
        # epsilon = max(0.01, 0.1 - 0.01*(self.num_epi/10))
        epsilon = max(0.01, 0.1 - 0.01 * (self.num_epi / 100))

        qval = self.q_net.forward(sub_state)
        coin = random.random()
        # Exploration.
        if coin < epsilon:
            dest_edge_id = random.randint(0, NUM_EDGE_DC)
        # Exploitation.
        else:
            # dest_edge_id = np.argmax(qval.detach().numpy())
            dest_edge_id = qval.detach().argmax().item()

        return dest_edge_id

        # # FIXME: dest id별로 exploration 또는 전체 exploration? 일단 전자
        # qval = self.q_net.forward(state)
        # dest_edge_ids = []
        # for i in range(qval.shape[0]):
        #     coin = random.random()
        #     # Exploration.
        #     if coin < epsilon:
        #         dest_edge_id = random.randint(0, 15)
        #     # Exploitation.
        #     else:
        #         # dest_edge_id = np.argmax(qval[i].detach().numpy())
        #         dest_edge_id = qval[i].argmax().item()
        #     dest_edge_ids.append(dest_edge_id)
        #
        # return dest_edge_ids


class Net(nn.Module):
    def __init__(self, dim_mig_nn_input):
        super(Net, self).__init__()
        self.input = nn.Linear(dim_mig_nn_input, NUM_NEURONS)
        self.hidden1 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.hidden2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        # FIXME:
        # self.output = nn.Linear(NUM_NEURONS, NUM_EDGE_DC + 1)
        self.output = nn.Linear(NUM_NEURONS, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.output(x)
        return x


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_CAPACITY)

    def put(self, transition):
        self.buffer.append(transition)

    def size(self):
        return len(self.buffer)
