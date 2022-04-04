import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 32
# learning_rate: 0.001~2 (DDPG 논문), 0.01 (도영 DQN), 0.0001 (Cartpole)
LEARNING_RATE = 0.0001
THRESHOLD_GRAD_NORM = 1
GAMMA = 0.98

# https://doheejin.github.io/pytorch/2021/09/22/pytorch-autograd-detect-anomaly.html
# torch.autograd.set_detect_anomaly(True)

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)


class ActorCriticv2MigrationAgent:
    def __init__(self, dim_mig_nn_input):
        self.net = Net(dim_mig_nn_input)
        self.num_epi = 0
        self.data = []

    def put_data(self, item):
        self.data.append(item)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            # done_mask_lst.append([done_mask])

        s_batch = torch.cat(s_lst)
        a_batch = torch.tensor(a_lst, dtype=torch.int64)
        r_batch = torch.tensor(r_lst, dtype=torch.float32)
        s_prime_batch = torch.cat(s_prime_lst)
        # s_batch = torch.tensor(s_lst, dtype=torch.float32)
        # a_batch = torch.tensor(a_lst, dtype=torch.int64)
        # r_batch = torch.tensor(r_lst, dtype=torch.float32)
        # s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float32)

        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch

    # FIXME: add comments.
    def get_action(self, state):
        fitness_scores = self.net.pi(state)

        # probs = torch.unsqueeze(torch.squeeze(fitness_scores, dim=1), dim=0)
        probs = fitness_scores.transpose(0, 1)
        dest_machine_index = Categorical(probs=probs).sample().item()

        return dest_machine_index, fitness_scores[dest_machine_index]

    def train(self):
        s, a, r, s_prime = self.make_batch()
        td_target = r + GAMMA * self.net.v(s_prime)
        delta = td_target - self.net.v(s)

        # pi = self.net.pi(s, softmax_dim=1)
        pi = self.net.pi(s, softmax_dim=0)
        # pi_a = pi.gather(1, a)
        pi_a = pi.gather(0, a)
        # loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.net.v(s), td_target.detach())
        pi_utilization_func = -torch.log(pi_a) * delta.detach()
        v_loss_func = F.smooth_l1_loss(self.net.v(s), td_target.detach())
        loss = pi_utilization_func + v_loss_func

        self.net.optimizer.zero_grad()
        # https://velog.io/@0hye/PyTorch-Nan-Loss-%EA%B2%80%EC%B6%9C-%EB%B0%A9%EB%B2%95
        if not torch.isfinite(loss.nanmean()):
            print('WARNING: non-finite loss, ending training ')
            exit(1)
        loss.nanmean().backward()
        # How to check norm of gradients?
        # https://discuss.pytorch.org/t/how-to-check-norm-of-gradients/13795/2
        # for p in list(filter(lambda p: p.grad is not None, self.net.parameters())):
        #     print(p.grad.detach().data.norm(2).item())

        # Gradient clipping to avoid nan loss.
        # https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-6/05-gradient-clipping
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=THRESHOLD_GRAD_NORM)
        self.net.optimizer.step()


class Net(nn.Module):
    def __init__(self, dim_mig_nn_input):
        super(Net, self).__init__()

        # Common
        self.input = nn.Linear(dim_mig_nn_input, NUM_NEURONS)
        self.hidden1 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.hidden2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)

        # Actor
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