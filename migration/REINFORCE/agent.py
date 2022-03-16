import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 64
# learning_rate: 0.001~2 (DDPG 논문), 0.01 (도영 DQN), 0.0001 (Cartpole)
LEARNING_RATE = 0.001
GAMMA = 0.98


class REINFORCEMigrationAgent:
    def __init__(self, dim_mig_nn_input, num_epi):
        self.net = Net(dim_mig_nn_input)
        self.num_epi = num_epi
        self.data = []

    def put_data(self, item):
        self.data.append(item)

    def get_action(self, state):
        # Produce fitness values of (M, s_i) pairs.
        fitness_values = self.net.forward(state)

        # Categorical converts items in probs into each relative probability sum to 1
        # e.g. Categorical([10, 20, 70]) => [0.1, 0.2, 0.7]
        probs = torch.unsqueeze(torch.squeeze(fitness_values, dim=1), dim=0)
        pair_index = Categorical(probs=probs).sample().item()

        # Returns the sampled (m_j, s_i) pair index and its probability.
        # Action: migrate s_i from s_i.machine to m_j.
        return pair_index, fitness_values[pair_index]

    # FIXME: trajectory 정보 main 인자로 넘기지 말고 agent 내부 변수로 처리하도록 수정
    # https://github.com/seungeunrho/minimalRL/blob/master/REINFORCE.py
    def train(self):
        R = 0
        self.net.optimizer.zero_grad()

        for r, prob in self.data[::-1]:
            R = r + GAMMA * R
            loss = -torch.log(prob) * R
            loss.backward()

        self.net.optimizer.step()
        self.data = []


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
        x = F.softmax(self.output(x), dim=0)
        return x
