import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# num_neurons: 30 (DDPG 논문), 128 (도영 DQN)
NUM_NEURONS = 32
# learning_rate: 0.001~2 (DDPG 논문), 0.01 (도영 DQN), 0.0001 (Cartpole)
LEARNING_RATE = 0.0001
# https://www.reddit.com/r/MachineLearning/comments/3n8g28/gradient_clipping_what_are_good_values_to_clip_at/
CLIPPING_NORM = 1
GAMMA = 0.98


class ActorCriticMigrationAgent:
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
            a_lst.append(a)
            # TODO: may scale reward for learning performance.
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # s_batch, a_batch, r_batch, s_prime_batch = \
        #     torch.cat(s_lst), torch.cat(a_lst), torch.tensor(r_lst).transpose(1, 0), torch.cat(s_prime_lst)
        s_batch = torch.cat(s_lst)
        a_batch = torch.cat(a_lst)
        r_batch = torch.tensor(r_lst).transpose(1, 0)
        s_prime_batch = torch.cat(s_prime_lst)

        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch

    # FIXME: fix the comments.
    def get_action(self, state):
        # Produce fitness scores of (M, s_i) pairs.
        fitness_scores = self.net.pi(state)

        # https://sanghyu.tistory.com/3
        probs = fitness_scores.permute(2, 0, 1)
        # Categorical converts items in probs into each relative probability sum to 1
        # e.g. Categorical([10, 20, 70]) => [0.1, 0.2, 0.7]
        sample_indices = Categorical(probs=probs).sample().numpy()[0].tolist()

        # Return the sampled index of (m_j, s_i) pair and its probability to be sampled.
        # Action: migrate s_i from s_i.machine to m_j.
        sample_probs = []
        for i in range(len(sample_indices)):
            sample_probs.append(fitness_scores[i][sample_indices[i]])

        return sample_indices, sample_probs

    def train(self):
        s, a, r, s_prime = self.make_batch()
        td_target = r + GAMMA * self.net.v(s_prime)
        delta = td_target - self.net.v(s)

        pi = self.net.pi(s)
        pi_a = pi.gather(1, a)
        # loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.net.v(s), td_target.detach())
        pi_utilization_func = -torch.log(pi_a) * delta.detach()
        v_loss_func = F.smooth_l1_loss(self.net.v(s), td_target.detach())
        loss = pi_utilization_func + v_loss_func

        self.net.optimizer.zero_grad()
        loss.mean().backward()
        # Gradient clipping to avoid nan loss.
        # https://discuss.pytorch.org/t/nan-loss-coming-after-some-time/11568
        # https://kh-kim.gitbook.io/natural-language-processing-with-pytorch/00-cover-6/05-gradient-clipping
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=CLIPPING_NORM)
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

    def pi(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        prob = F.softmax(self.fc_pi(x), dim=0)
        return prob

    def v(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))

        v = self.fc_v(x)
        return v
