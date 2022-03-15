import collections
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 64
# learning_rate: 0.001~2 (DDPG 논문), 0.01 (도영 DQN), 0.0001 (Cartpole)
LEARNING_RATE = 0.001
# replay mem capacity: 10000 (Cartpole), 2500 (도영 DQN)
# FIXME: 1 episode에 대략 몇 개 저장되는지 확인 후 적절한 값 설정
BUFFER_CAPACITY = 10000
# FIXME:
LEAST_SIZE_TO_LEARN = 1000
# FIXME:
# mini-batch sampling size: 32 (Cartpole), 16 (도영)
BATCH_SIZE = 16
GAMMA = 0.98


class DQNv2MigrationAgent:
    def __init__(self, dim_mig_nn_input, num_epi):
        self.q_net = Net(dim_mig_nn_input)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.target_q_net = Net(dim_mig_nn_input)
        self.update_target_q_function()

        self.num_epi = num_epi
        self.memory = ReplayBuffer()

    def get_action(self, state):
        # Linear annealing from 10% to 1%.
        epsilon = max(0.01, 0.1 - 0.01*(self.num_epi/100))

        # coin = random.random()
        coin = np.random.uniform(0, 1)
        if coin < epsilon:
            # return random.randint(0, 1)
            # return torch.LongTensor([random.randint(0, state.shape[0]-1)])
            return random.randint(0, state.shape[0]-1)
        else:
            return self.q_net.forward(state).argmax().item()

    def memorize(self, state, action, reward, state_next, done_mask):
        # TODO: scale reward if needed for performance (e.g, reward/100.0).
        scaled_reward = reward
        self.memory.put((state, action, scaled_reward, state_next, done_mask))

    def train(self):
        # FIXME: 한 번 호출에 몇 번의 q_net 업데이트 수행? v1: 1번, v2: 10번
        # for i in range(10):
        if self.memory.size() > LEAST_SIZE_TO_LEARN:
            # s, a, r, s_prime, done_mask = self.memory.sample(BATCH_SIZE)
            s, a, r, s_prime = self.memory.sample(BATCH_SIZE)

            q_out = self.q_net(s)
            # q_a = q_out.gather(1, a)
            q_a = q_out.gather(0, a)
            # max_q_prime = self.target_q_net(s_prime).max(1)[0].unsqueeze(1)
            max_q_prime = self.target_q_net(s_prime).max(0)[1]
            # target = r + GAMMA * max_q_prime * done_mask
            target = r + GAMMA * max_q_prime
            loss = F.smooth_l1_loss(q_a, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_q_function(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())


class Net(nn.Module):
    def __init__(self, dim_mig_nn_input):
        super(Net, self).__init__()
        self.input = nn.Linear(dim_mig_nn_input, NUM_NEURONS)
        self.hidden1 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.hidden2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.output = nn.Linear(NUM_NEURONS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

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

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
        #        torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
        #        torch.tensor(done_mask_lst)
        return torch.cat(s_lst), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.cat(s_prime_lst)

    def size(self):
        return len(self.buffer)
