import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Categorical

# ! 학습 feature 중 machine-service 간 path cost에 임의로 가중치 부여 (학습 잘되도록)
PATH_COST_WEIGHT = 10


class Transition:
    def __init__(self, observation, action, reward, clock):
        # FIXME: state? observation?
        self.observation = observation
        self.action = action
        self.reward = reward
        self.clock = clock


class REINFORCEAlgorithm:
    def __init__(self, agent, reward_giver, features_normalize_func, features_extract_func):
        self.agent = agent
        self.reward_giver = reward_giver
        # self.features_normalize_func = features_normalize_func
        # self.features_extract_func = features_extract_func
        # FIXME: use agent.data or trajectory instead
        self.current_trajectory = []
        # for debug purpose
        self.selected_pairs = []

    def extract_machine_features(self, machine):
        return [machine.cpu, machine.memory, machine.disk]

    def extract_service_features(self, service):
        return [service.service_profile.cpu, service.service_profile.memory,
                service.service_profile.disk, service.service_profile.duration]

    def make_batch(self, valid_pairs):
        feature_vectors = []
        for machine, service in valid_pairs:
            # !convert a valid pair to [machine.profile + service.profile + path cost] as a feature vector
            path_cost = [PATH_COST_WEIGHT * machine.mec_net.get_path_cost(service.edge_machine_id, machine.id)]
            feature_vector = self.extract_machine_features(machine) + self.extract_service_features(service) + path_cost
            feature_vectors.append(feature_vector)

        return feature_vectors
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # return scaler.fit_transform(feature_vectors)

    def __call__(self, mec_net, clock):
        # a pending service request is waiting for being deployed on a machine as a service instance
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines

        # !step 1: find all possible (m, s) pairs
        valid_machine_service_pairs = []
        for service in waiting_services:
            for machine in machines:
                if machine.can_accommodate(service.service_profile):
                    valid_machine_service_pairs.append((machine, service))

        # !step 2: find one (m, s) pair with higher probability (as fitness value)
        # !(not always the highest one due to the behavior of Categorical below)
        if len(valid_machine_service_pairs) != 0:
            # convert (m, s) pairs to feature vectors (mini-batch)
            feature_vectors = self.make_batch(valid_machine_service_pairs)
            state = torch.from_numpy(np.array(feature_vectors).astype(float)).float()
            # produce fitness values for each (machine, service) pair
            fitness_values = self.agent.pi(state)

            # Categorical converts items in probs into each relative probability sum to 1
            # e.g. Categorical([10, 20, 70]) => [0.1, 0.2, 0.7]
            probs = torch.unsqueeze(torch.squeeze(fitness_values, dim=1), dim=0)
            pair_index = Categorical(probs=probs).sample().item()

            # FIXME: use reward_giver
            # use path cost of the selected (machine, service (its edge)) pair as immediate reward
            # default = 1 if edge machine is selected
            # else reward decreases as path cost increases
            # !엄밀하게는 해당 서비스의 duration까지 고려해서 계산해야됨 (accumulated path cost for this service deployment)
            # !일단은 테스팅 목적으로 단순하게 계산함 (duration 보장 안되는 고장 상황에서는 이게 나을수도?)
            path_cost = feature_vectors[pair_index][-1] / PATH_COST_WEIGHT
            reward = 1
            if path_cost != 0:
                reward = 1 / path_cost

            # !Transition(s, a, r)
            # !s: valid (m, s) pairs (each pair translated to [m.profile, s.profile, path_cost]) at this scheduling tick
            # !a: probability of selecting the (m, s) pair (fitness value of the (m, s) pair)
            transition = Transition(state, fitness_values[pair_index], reward, clock)
            self.current_trajectory.append(transition)
            self.selected_pairs.append(valid_machine_service_pairs[pair_index])

            return valid_machine_service_pairs[pair_index]

        # this represents the end of deployment scheduling at this tick
        else:
            self.current_trajectory.append(Transition(None, None, None, clock))
            self.selected_pairs.append(None)

        return None, None
