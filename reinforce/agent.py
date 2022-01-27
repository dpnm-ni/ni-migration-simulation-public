import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Hyperparameters
learning_rate = 0.0002
gamma = 0.98


class TDActorCritic(nn.Module):
    def __init__(self, dim_nn_input):
        super(TDActorCritic, self).__init__()
        self.data = []

        # # 첫번째 layer: 정책넷과 밸류넷 공통
        # self.fc1 = nn.Linear(dim_nn_input, 256)
        #
        # # 두번째 layer: 정책넷(Actor) 고유
        # # output: prob(move_left), prob(move_right)
        # # self.fc_pi = nn.Linear(256, 2)
        # # output: fitness value of each (machine, service) pair
        # self.fc_pi = nn.Linear(256, 1)
        #
        # # 두번째 layer: 밸류넷(Critic) 고유
        # # output: V(s)
        # self.fc_v = nn.Linear(256, 1)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # # 첫번째 layer: 정책넷과 밸류넷 공통
        # self.fc1 = nn.Linear(dim_nn_input, 3)
        #
        # self.pi1 = nn.Linear(3, 9)
        # self.pi2 = nn.Linear(9, 18)
        # self.pi3 = nn.Linear(18, 9)
        # self.pi4 = nn.Linear(9, 1)
        #
        # self.v1 = nn.Linear(256, 1)

        # self.fc1 = nn.Linear(dim_nn_input, 128)
        # self.fc2 = nn.Linear(128, 1)

        self.fc1 = nn.Linear(dim_nn_input, 3)
        self.fc2 = nn.Linear(3, 9)
        self.fc3 = nn.Linear(9, 18)
        self.fc4 = nn.Linear(18, 9)
        self.fc5 = nn.Linear(9, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # 정책넷 forward 함수
    def pi(self, x):
        # x = self.fc1(x)
        #
        # x = F.tanh(self.pi1(x))
        # x = F.tanh(self.pi2(x))
        # x = F.tanh(self.pi3(x))
        # x = self.pi4(x)
        #
        # return x

        # x = F.relu(self.fc1(x))
        # x = F.softmax(self.fc2(x), dim=0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x), dim=0)

        return x

    # 밸류넷 forward 함수
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r / 100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = \
            torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
            torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
            torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self, observations, actions, rewards):
        R = 0
        self.optimizer.zero_grad()

        # # G_t = r_t+1 + gamma * G_t+1 점화식 이용하여 뒤에서부터(G_t+1) 앞으로(G_t) 업데이트 (cf. MCControl cum_reward)
        # for r, prob in self.data[::-1]:
        #     R = r + gamma * R
        #     # 중요: 여기서 정의하는 loss 함수 J(theta)=E[sigma(Rt)]는 미분했을 때 policy gradient 즉 log(PI(s,a)) * Q(s,a)가 되는 함수 (교재 222p)
        #     # R(Gt)은 Q_true의 불편추정량 즉 Q_true(s,a) = E[R] 이므로 REINFORCE에서는 Q대신 R을 사용하며, -1을 곱해서 GD 대신 GA 수행
        #     loss = -torch.log(prob) * R
        #     loss.backward()  # loss function에 대한 그라디언트 계산 => policy gradient
        # # 중요: loss function 즉 J(theta)의 정의(정책 평가 함수)에 따라 loss를 늘리는 방향(더 큰 R을 내는 방향)으로 정책넷의 파라미터 업데이트 (그라디언트 어센트)
        # self.optimizer.step()
        # self.data = []

        data = [(reward, prob_a) for reward, prob_a in zip(rewards, actions)]
        for r, pi_a in data[::-1]:
            if pi_a is not None:
                R = r + gamma * R
                loss = -torch.log(pi_a) * R
                loss.backward()
        self.optimizer.step()





    # def train_net(self, observations, actions, rewards):
    #     # s, a, r, s_prime, done = self.make_batch()
    #     # s_batch = torch.tensor(observations, dtype=torch.float)
    #     # s_batch = torch.from_numpy(observations).float()
    #     # s_batch = torch.tensor(observations, dtype=torch.float)
    #     s_batch = torch.from_numpy(np.vstack(np.array(observations))).float()
    #     # s_batch = torch.tensor([torch.from_numpy(observation).float() for observation in observations])
    #     # a_batch = torch.tensor(actions)
    #     # a_batch = [torch.from_numpy(action) for action in actions]
    #     a_batch = actions
    #     # r_batch = torch.tensor(rewards, dtype=torch.float)
    #     # r_batch = [torch.tensor(reward).float() for reward in rewards]
    #     r_batch = torch.tensor(rewards, dtype=torch.float)
    #
    #     # TD target (R + gamma * V(s'))은 V_true에 대한 추정량
    #     # td_target = r + gamma * self.v(s_prime) * done
    #     td_target = r_batch + gamma * self.v(s_batch)       # note v(s_batch) not s_prime_batch
    #     # delta = td_target - self.v(s)  # TD error 계산
    #     delta = td_target - self.v(s_batch)
    #
    #     # pi = self.pi(s, softmax_dim=1)
    #     pi = self.pi(s_batch, softmax_dim=1)
    #
    #     # pi_a = pi.gather(1, a)
    #     pi_a = pi.gather(1, a_batch)
    #     # pi_a = []
    #     # for observation, action in observations, actions:
    #     #     if action is not None:
    #     #         observation
    #
    #
    #     # 중요: 여기서 정의하는 loss 함수는 1.정책넷(actor)에 대한 평가 함수 J(theta)와 2.밸류넷(critic)에 대한 손실 함수 L(w)의 합으로 구성됨 (구현 이슈)
    #     # J(theta)는 미분했을 때 policy gradient => log(PI(s,a)) * Q(s,a) = log(PI(s,a)) * delta (교재 246p)
    #     # loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
    #     loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s_batch), td_target.detach())
    #
    #     self.optimizer.zero_grad()
    #     loss.mean().backward()  # 10개 transition의 평균 loss에 대한 그라디언트 계산 => policy gradient + grad(L(w))
    #     # 원칙: 1번항(정책넷 평가 함수)은 maximize 되어야 하고 2번항(밸류넷 손실 함수)은 minimize 되어야 함
    #     # loss를 2번항 - 1번항 즉 두 항의 차로 보고 그라디언트 디센트 적용하면, 1번항은 커지고(0에 가깝게) 2번항은 작아지는 방향으로
    #     # 각각의 파라미터 theta와 w가 업데이트 되어 원칙에 부합
    #     self.optimizer.step()
