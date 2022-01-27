import numpy as np
import torch

from torch.distributions import Categorical


# TODO: next_observation (s') is omitted.
#  what will be s' in this scheduling problem and learning it in training will result in better perf?
class Node(object):
    def __init__(self, observation, action, reward, clock):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class RLAlgorithm(object):
    def __init__(self, agent, reward_giver, features_normalize_func, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        self.features_normalize_func = features_normalize_func
        self.features_extract_func = features_extract_func
        # FIXME: use agent.data or trajectory instead
        self.current_trajectory = []

    def extract_features(self, valid_pairs):
        features = []
        for machine, service in valid_pairs:
            features.append([machine.cpu, machine.memory, machine.disk] + self.features_extract_func(service))
        features = self.features_normalize_func(features)
        return features

    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        valid_machine_service_pairs = []

        for service in waiting_services:
            for machine in machines:
                if machine.can_accommodate(service.service_profile):
                    valid_machine_service_pairs.append((machine, service))

        # fall into None action that means the end of this scheduling time step. see section 3.2.3 in the DeepJS paper
        if len(valid_machine_service_pairs) == 0:
            # note that r is self.reward_giver.get_reward()
            # self.current_trajectory.append(Node(None, None, self.reward_giver.get_reward(), clock))
            # features = np.array([[0, 0, 0, 0, 0, 0, 0]])
            features = None
            # features = np.array([None, None, None, None, None, None, None])
            self.current_trajectory.append(Node(features, None, self.reward_giver.get_reward(), clock))
            return None, None
        else:
            # TODO: double check the function logic (now normalization func turned off)
            features = self.extract_features(valid_machine_service_pairs)
            # features = tf.convert_to_tensor(features, dtype=np.float32)
            # after extract_features, features should be numpy object
            # features = torch.from_numpy(features).float()
            # logits = self.agent.brain(features)
            pi_input = torch.from_numpy(features).float()
            # pi_net produces fitness values for each (machine, service) pair
            prob = self.agent.pi(pi_input)
            prob1 = torch.unsqueeze(torch.squeeze(prob, dim=1), dim=0)

            # pair_index = tf.squeeze(tf.multinomial(logits, num_samples=1), axis=1).numpy()[0]
            # https://stackoverflow.com/a/51194665/5204099
            # pair_index = Categorical(logits=prob).sample().item()
            pair_index = Categorical(prob1).sample().item()

            # Node(s, a, r, clock)
            # note that reward r is 0, not self.reward_giver.get_reward()
            # node = Node(features, pair_index, 0, clock)
            node = Node(features, prob[pair_index], 0, clock)
            self.current_trajectory.append(node)

        return valid_machine_service_pairs[pair_index]
