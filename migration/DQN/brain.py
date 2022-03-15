import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from migration.DQN.replay_memory import ReplayMemory, Transition

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 32
# learning_rate: 0.001~2 (DDPG 논문), 0.01 (도영 DQN), 0.0001 (Cartpole)
LEARNING_RATE = 0.01
# replay mem capacity: 10000 (Cartpole), 2500 (도영 DQN)
BUFFER_CAPACITY = 2500
# mini-batch sampling size: 32 (Cartpole), 16 (도영)
BATCH_SIZE = 1
GAMMA = 0.98


class DQNMigrationBrain:
    def __init__(self, dim_mig_nn_input):
        self.memory = ReplayMemory(BUFFER_CAPACITY)

        # !input: [m.cpu, m.mem, m.disk, s.cpu, s.mem, s.disk, s.dur, m-s.path_cost, ...]
        # !inputs: input * # of (machine, service) pairs at a mig decision tick => inputs 자체로 이미 미니배치
        # FIXME:
        #  transition(s, a, r, s') 단위가 inputs(s: placement pairs per tick)가 되도록 수정할 것 (이렇게 해야 정석임)
        #  1. 기존 DQN 코드 가져와서 그대로 쓴다면 BATCH_SIZE 값을 1로 설정
        #  2. state 자체를 inputs: (M, S) pairs => input: [all M, S]로 변경
        #  => 이러면 step-wise 형태로 DQN에 적합해 보이는데 알고리즘/구조 전체적으로 바꿔야 되고 DRL 학습 자체가 잘 될지 의문
        self.main_q_net = Net(dim_mig_nn_input)
        self.target_q_net = Net(dim_mig_nn_input)

        # 최적화 기법 선택. main net만 gradient로 학습시키고 target net에 주기적으로 overwrite
        self.optimizer = optim.Adam(self.main_q_net.parameters(), lr=LEARNING_RATE)

    def replay(self):
        '''Experience Replay로 신경망의 결합 가중치 학습'''

        # 1. 저장된 transition의 수를 확인
        if len(self.memory) < BATCH_SIZE:
            return

        # 2. 미니배치 생성
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3. 정답신호로 사용할 Q(s_t, a_t)를 계산
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4. 결합 가중치 수정
        self.update_main_q_network()

    def decide_action(self, state, episode):
        '''현재 상태로부터 행동을 결정함'''
        # ε-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다
        epsilon = 0.5 * (1 / (episode + 1))

        # !epsilon = Prob(exploration)
        # !episode 증가해서 epsilon 작아지면 exploration 비중 줄어들어야 함
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_net.eval()  # 신경망을 추론 모드로 전환
            with torch.no_grad():
                # !참고: REINFORCE version
                # !policy gradient에서는 정책함수 pi의 확률분포에 따라 action을 샘플링하고 pi(s,a) 즉 prob(a|s) 값을 직접 학습함
                # !but DQN에서는 e-greedy 통해 샘플링하므로 action 즉 max로 만드는 (M, s_i) pair의 인덱스만 뽑으면 됨
                # fitness_values = self.main_q_net(state)
                # Categorical converts items in probs into each relative probability sum to 1
                # e.g. Categorical([10, 20, 70]) => [0.1, 0.2, 0.7]
                # probs = torch.unsqueeze(torch.squeeze(fitness_values, dim=1), dim=0)
                # pair_index = Categorical(probs=probs).sample().item()
                # action = pair_index

                # !어프로치: score[action] - score[current] > threshold 일때만 migration 수행하여 빈번한 migration 방지
                # !threshold 최적값 찾는것 자체가 별도의 문제. 일단 argmax action 무조건 리턴
                # action = self.main_q_net(state).max(1)[1].view(1, 1)
                action = self.main_q_net(state).max(0)[1].view(1, 1)
                # 신경망 출력의 최댓값에 대한 인덱스 = max(1)[1]
                # .view(1,1)은 [torch.LongTensor of size 1] 을 size 1*1로 변환하는 역할을 한다

        else:
            # 행동을 무작위로 반환(0 혹은 1)
            # action = torch.LongTensor(
            #     [[random.randrange(self.num_actions)]])  # 행동을 무작위로 반환(0 혹은 1)
            action = torch.LongTensor(
                [[random.randrange(state.shape[0])]])  # 행동을 무작위로 반환(0 혹은 1)
            # action은 [torch.LongTensor of size 1*1] 형태가 된다

        return action

    def make_minibatch(self):
        '''2. 미니배치 생성'''

        # 2.1 메모리 객체에서 미니배치를 추출
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 각 변수를 미니배치에 맞는 형태로 변형
        # transitions는 각 단계 별로 (state, action, state_next, reward) 형태로 BATCH_SIZE 갯수만큼 저장됨
        # 다시 말해, (state, action, state_next, reward) * BATCH_SIZE 형태가 된다
        # 이것을 미니배치로 만들기 위해
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE) 형태로 변환한다
        batch = Transition(*zip(*transitions))

        # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있도록 Variable로 만든다
        # state를 예로 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 갯수만큼 있는 형태이다
        # 이를 torch.FloatTensor of size BATCH_SIZE*4 형태로 변형한다
        # 상태, 행동, 보상, non_final 상태로 된 미니배치를 나타내는 Variable을 생성
        # cat은 Concatenates(연접)을 의미한다
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''정답신호로 사용할 Q(s_t, a_t)를 계산'''

        # 3.1 신경망을 추론 모드로 전환
        self.main_q_net.eval()
        self.target_q_net.eval()

        # 3.2 신경망으로 Q(s_t, a_t)를 계산
        # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력하며
        # [torch.FloatTensor of size BATCH_SIZEx2] 형태이다
        # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가
        # 왼쪽이냐 오른쪽이냐에 대한 인덱스를 구하고, 이에 대한 Q값을 gather 메서드로 모아온다
        # self.state_action_values = self.main_q_net(self.state_batch).gather(1, self.action_batch)
        self.state_action_values = self.main_q_net(self.state_batch).gather(0, self.action_batch)

        # 3.3 max{Q(s_t+1, a)}값을 계산한다 이때 다음 상태가 존재하는지에 주의해야 한다

        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듬
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state)))
        # 먼저 전체를 0으로 초기화
        next_state_values = torch.zeros(BATCH_SIZE)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
        # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함 ([0]이면 값. DQN brain 코드 참조)
        # a_m[non_final_mask] = self.main_q_net(self.non_final_next_states).detach().max(1)[1]
        a_m[non_final_mask] = self.main_q_net(self.non_final_next_states).detach().max(0)[1]

        # 다음 상태가 있는 것만을 걸러내고, size 32를 32*1로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
        # detach() 메서드로 값을 꺼내옴
        # squeeze() 메서드로 size[minibatch*1]을 [minibatch]로 변환
        # !main net의 Q(s,a)를 최대로하는 a_m을 구해 target net에서의 Q(s, a_m)을 구함
        # next_state_values[non_final_mask] = self.target_q_net(
        #     self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()
        next_state_values[non_final_mask] = self.target_q_net(
            self.non_final_next_states).gather(0, a_m_non_final_next_states).detach().squeeze()

        # 3.4 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산한다
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        '''4. 결합 가중치 수정'''

        # 4.1 신경망을 학습 모드로 전환
        self.main_q_net.train()

        # 4.2 손실함수를 계산 (smooth_l1_loss는 Huber 함수)
        # expected_state_action_values은
        # size가 [minibatch]이므로 unsqueeze하여 [minibatch*1]로 만든다
        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))

        # 4.3 결합 가중치를 수정한다
        self.optimizer.zero_grad()  # 경사를 초기화
        loss.backward()  # 역전파 계산
        self.optimizer.step()  # 결합 가중치 수정

    def update_target_q_network(self):  # DDQN에서 추가됨
        '''Target Q-Network을 Main Q-Network와 맞춤'''
        self.target_q_net.load_state_dict(self.main_q_net.state_dict())


# 신경망 구성
class Net(nn.Module):
    def __init__(self, dim_mig_nn_input):
        super(Net, self).__init__()
        self.input = nn.Linear(dim_mig_nn_input, NUM_NEURONS)
        self.hidden1 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.hidden2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.output = nn.Linear(NUM_NEURONS, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        # x = F.softmax(self.output(x), dim=0)
        x = self.output(x)
        return x
