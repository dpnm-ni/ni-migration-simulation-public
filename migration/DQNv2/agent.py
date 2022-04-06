import collections
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 32
# learning_rate: 0.001~2 (DDPG 논문), 0.01 (도영 DQN), 0.0001 (Cartpole)
LEARNING_RATE = 0.001
THRESHOLD_GRAD_NORM = 1
# replay mem capacity: 10000 (Cartpole), 2500 (도영 DQN)
# FIXME: 1 episode에 대략 몇 개 저장되는지 확인 후 적절한 값 설정
BUFFER_CAPACITY = 2500
# FIXME: 1 에피소드에 각 엣지별 평균 50~100개 transition 생성됨. 에피소드 #2부터 학습 시작하도록
LEAST_SIZE_TO_LEARN = 100
# mini-batch sampling size: 32 (Cartpole), 16 (도영)
BATCH_SIZE = 16
GAMMA = 0.98


class DQNv2MigrationAgent:
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

    def get_action_set(self, state):
        # Linear annealing from 10% to 1% (when num_epi is 100).
        # epsilon = max(0.01, 0.1 - 0.01*(self.num_epi/10))
        epsilon = max(0.01, 0.1 - 0.01 * (self.num_epi / 100))

        # FIXME: dest id별로 exploration 또는 전체 exploration? 일단 전자
        qval = self.q_net.forward(state)
        dest_edge_ids = []
        for i in range(state.shape[0]):
            coin = random.random()
            # Exploration.
            if coin < epsilon:
                dest_edge_id = random.randint(0, 15)
            # Exploitation.
            else:
                dest_edge_id = np.argmax(qval[i].detach().numpy())
            dest_edge_ids.append(dest_edge_id)

        return dest_edge_ids

    def memorize(self, state, action, reward, state_next):
        # TODO: scale reward if needed for performance (e.g, reward/100.0).
        scaled_reward = reward
        self.memory.put((state, action, scaled_reward, state_next))

    def train(self):
        # FIXME: 한 번 호출에 몇 번의 q_net 업데이트 수행? v1: 1번, v2: 10번
        # for i in range(10):
        if self.memory.size() > LEAST_SIZE_TO_LEARN:
            mini_batch = random.sample(self.memory.buffer, BATCH_SIZE)
            for transition in mini_batch:
                s, a, r, s_prime = transition

                # max a s.t max Q(s, a)
                q_a = self.q_net(s).gather(1, a.view(-1, 1))
                max_q_prime = self.target_q_net(s_prime).max(1)[0].unsqueeze(1)
                target = r + GAMMA * max_q_prime
                loss = F.smooth_l1_loss(q_a, target)

                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=THRESHOLD_GRAD_NORM)
                self.optimizer.step()

    def update_target_q_function(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())


class Net(nn.Module):
    def __init__(self, dim_mig_nn_input):
        super(Net, self).__init__()
        self.input = nn.Linear(dim_mig_nn_input, NUM_NEURONS)
        self.hidden1 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.hidden2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        # FIXME:
        self.output = nn.Linear(NUM_NEURONS, 16)

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
