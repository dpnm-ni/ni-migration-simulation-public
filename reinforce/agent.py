import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 30
# learning_rate: 0.002, 0.001 (DDPG 논문) ~ 0.01 (도영 DQN)
LEARNING_RATE = 0.001
GAMMA = 0.98


class REINFORCEAgent(nn.Module):
    def __init__(self, dim_nn_input):
        super(REINFORCEAgent, self).__init__()

        # !DeepJS section 3.2.4에 따르면 현재 스케쥴링 틱의 (machine, service) pair 리스트(대응하는 feature vectors)를
        # !batch로 입력했을 때 각 (machine, service) pair의 fitness value가 계산되어 output으로 출력됨
        # !=> 각 (m,s) pair의 fitness value를 담고 있는 output list는 확률 분포로 간주되어 가장 높은 값의 (m,s) pair가 선택될 확률 높음
        # !=> DRL 통해 해당 뉴럴넷이 최적의 pair(뭔지 모름)에 대해 더 높은 fitness value를 출력하도록 학습 시키는게 목적
        # input: [m.cpu, m.mem, m.disk, s.cpu, s.mem, s.disk, s.dur, m-s.path_cost]
        self.input = nn.Linear(dim_nn_input, NUM_NEURONS)
        self.hidden1 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.hidden2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.output = nn.Linear(NUM_NEURONS, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)

    # 정책넷 forward 함수
    def pi(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x), dim=0)
        # x = self.output(x)
        return x

    # FIXME: trajectory 정보 main 인자로 넘기지 말고 agent 내부 변수로 처리하도록 수정
    # https://github.com/seungeunrho/minimalRL/blob/master/REINFORCE.py
    def train_net(self, observations, actions, rewards, episode):
        R = 0
        self.optimizer.zero_grad()

        # !actions[i]: 스케쥴링 틱 i에서의 deployment action 즉 선택된 (machine, service) pair의 fitness value 또는 None (action)
        # !=> 사용된 DRL 알고리즘에서 해당 pair가 선택될 확률(pi_a, prob_a)의 형태로 저장함
        data = [(reward, prob_a) for reward, prob_a in zip(rewards, actions)]
        for r, pi_a in data[::-1]:
            if pi_a is not None:
                R = r + GAMMA * R
                loss = -torch.log(pi_a) * R
                loss.backward()

        self.optimizer.step()
