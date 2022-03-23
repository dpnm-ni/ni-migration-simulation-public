import numpy as np
import torch
from core.algorithm import Algorithm


class REINFORCEMigrationAlgorithm(Algorithm):
    def __init__(self, agent, num_epi, reward_giver=None):
        self.agent = agent
        self.agent.num_epi = num_epi
        self.reward_giver = reward_giver

        # Valid at the current state only. Should be initialized before the next state.
        self.current_machine_service_pairs = None
        self.current_batch = None
        self.state = None

        # For debug.
        self.hist_reward_lat = []
        self.hist_reward_avail = []
        self.hist_reward_type = []
        self.hist_total_reward = []

    def make_input_batch(self, mec_net):
        batch = []
        for machine, service in self.current_machine_service_pairs:
            # feature_vector = [mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)] \
            #                  + [machine.cpu, machine.memory, machine.disk] \
            #                  + [machine.mon_disk_utilization, machine.mon_disk_overutil_cnt] \
            #                  + [service.service_profile.cpu, service.service_profile.memory, service.service_profile.disk,
            #                     service.service_profile.duration, service.service_profile.e2e_latency]
            feature_vector = [mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)] \
                             + [machine.cpu, machine.memory, machine.disk] \
                             + [machine.mon_disk_utilization] \
                             + [service.service_profile.cpu, service.service_profile.memory,
                                service.service_profile.disk,
                                service.service_profile.duration, service.service_profile.e2e_latency]

            batch.append(feature_vector)
        return np.array(batch).astype(float)

    def __call__(self, mec_net, service):
        action, prob_action = self.agent.get_action(self.state)

        # FIXME: use reward_giver.
        # Use the path cost of the selected (m_new, s_i) pair as a immediate reward.
        # The path cost is 0 if the machine is in the nearest edge DC to s_i.user_loc.
        # !엄밀하게는 해당 서비스의 duration까지 고려해서 계산해야됨 (expected accumulated path cost by this placement)
        # !일단은 테스팅 목적으로 단순하게 계산함 (duration 보장 안되는 고장 상황에서는 성능상 이게 나을수도?)
        path_cost = self.current_batch[action][0]
        dest_machine = self.current_machine_service_pairs[action][0]

        reward_service_latency = 1 / (path_cost + 1)

        # failure_score = self.current_batch[action][5]
        failure_score = self.current_batch[action][4]
        reward_service_availability = 1 / (failure_score + 1)

        reward_service_type = 1 \
            if service.get_service_type() == 3 and dest_machine.id == 0 \
            else 1 / (service.service_profile.e2e_latency - path_cost + 1)
        # reward_service_type = 1 / (service.service_profile.e2e_latency - path_cost + 1)

        # REWARD = 0.5 * reward_service_latency + 0.5 * reward_service_type
        REWARD = 1 / ((1 - reward_service_latency) + (1 - reward_service_type) + (1 - reward_service_availability) + 1e-1)

        # For debug.
        self.hist_reward_lat.append(reward_service_latency)
        self.hist_reward_avail.append(reward_service_availability)
        self.hist_reward_type.append(reward_service_type)
        self.hist_total_reward.append(REWARD)

        return dest_machine, prob_action, REWARD

    def get_state(self, mec_net, service):
        # Construct valid (M, s_i) pairs.
        machine_service_pairs = []
        for machine in mec_net.machines:
            # if self.can_satisfy_e2e_latency(mec_net, service, machine):
                if machine.can_accommodate(service.service_profile):
                    machine_service_pairs.append((machine, service))

        self.current_machine_service_pairs = machine_service_pairs
        if len(self.current_machine_service_pairs) != 0:
            # Ensure that the row dimensions of the batch and the pairs are same
            # so that the selected action falls into the same machine in __call__.
            self.current_batch = self.make_input_batch(mec_net)
            # state = torch.from_numpy(np.array(self.current_batch).astype(float)).float()
            self.state = torch.from_numpy(self.current_batch).float()
            return self.state
        # When no machines are available to the service for now.
        else:
            return None
