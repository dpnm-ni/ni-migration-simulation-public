import numpy as np
import torch
from torch.distributions import Categorical
from core.algorithm import Algorithm
from core.injector import DISK_FAULT_THRESHOLD
from base_logger import log
from decimal import Decimal


class DQNv2MigrationAlgorithm(Algorithm):
    def __init__(self, agent, num_epi, reward_giver=None):
        self.agent = agent
        # Set agent's num_epi for decaying epsilon.
        self.agent.num_epi = num_epi
        self.reward_giver = reward_giver

        # Valid at the current state only. Should be initialized before the next state.
        self.current_machine_service_pairs = None
        self.current_batch = None
        self.state = None

    # def compute_migration_cost(self, mec_net, service, source_machine, dest_machine):
    #     # Migration cost is zero in the case of current-to-current.
    #     if source_machine.id == dest_machine.id:
    #         return [0, 0]
    #     # TODO: need to define a refined model for migration cost.
    #     else:
    #         # !임시로 정의: total mig time (VM transmission cost) + mig service downtime (larger if write-intensive app)
    #         # !MT: (service.memory + service.disk) * #hops
    #         # !DT: service.memory / service.duration (=write intensity 또는 service.machine.mem_utilization / #services 등...)
    #         MT = (service.memory + service.disk) * mec_net.get_path_length(
    #             source_id=source_machine.machine_profile.edgeDC_id, dest_id=dest_machine.id)
    #         DT = service.memory / round(Decimal(service.duration), 9)
    #
    #         # !scale 다른데 normalization 필요? 가중치 필요?
    #         # return [MT + DT]
    #         return [MT, DT]
    #
    # # TODO: need to define a refined model for failure score prediction. DL?
    # def compute_failure_score(self, machine):
    #     return [machine.mon_disk_overutil_cnt]
    #
    # # machine.cpu: remaining cpu cores
    # def compute_resource_utilization(self, machine):
    #     return [1 - (machine.cpu/machine.cpu_capacity),
    #             1 - (machine.memory/machine.memory_capacity),
    #             1 - (machine.disk/machine.disk_capacity)]

    def make_input_batch(self, mec_net):
        batch = []
        for machine, service in self.current_machine_service_pairs:
            # # Convert a pair of (m_j, s_i) into the corresponding feature vector.
            # # We expect an output fitness score of (m_j, s_i) should be higher when the following values are lower:
            # # 1. path cost from s_i.user_loc (not s_i.machine) to m_j (-> service latency) [dim: 1]
            # # 2. migration cost from s_i.machine to m_j [dim: 2]
            # # 3. predicted failure score of m_j [dim: 1]
            # # 4. resource utilization of m_j (-> resource LB) [dim: 3]
            # feature_vector = [machine.mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)] \
            #                  + self.compute_migration_cost(mec_net, service, service.machine, machine)\
            #                  + self.compute_failure_score(machine) \
            #                  + self.compute_resource_utilization(machine)

            feature_vector = [mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)] \
                             + [machine.cpu, machine.memory, machine.disk] \
                             + [service.service_profile.cpu, service.service_profile.memory, service.service_profile.disk,
                                service.service_profile.duration, service.service_profile.e2e_latency]

            batch.append(feature_vector)
        return np.array(batch).astype(float)

    # !기존 서버에서 기존 서버로의 placement (no migration) 결과도 포함되어 있음에 유의
    # !=> 해당 케이스는 reward를 다르게 계산해야될 것으로 보이나 일단은 동일하게 취급 후 추후 세분화 예정
    def __call__(self, mec_net, service):
        # Return as an action a pair index that maximizes Q(state, action),
        # where the state is batched feature vectors of (M, s_i) pairs.
        action = self.agent.get_action(self.state)

        # FIXME: use reward_giver.
        # Use the path cost of the selected (m_new, s_i) pair as a immediate reward.
        # The path cost is 0 if the machine is in the nearest edge DC to s_i.user_loc.
        # !엄밀하게는 해당 서비스의 duration까지 고려해서 계산해야됨 (expected accumulated path cost by this placement)
        # !일단은 테스팅 목적으로 단순하게 계산함 (duration 보장 안되는 고장 상황에서는 성능상 이게 나을수도?)
        path_cost = self.current_batch[action][0]
        dest_machine = self.current_machine_service_pairs[action][0]

        # reward_service_latency = np.log(1 - (path_cost / mec_net.max_path_cost) + 1e-1) + 1
        # if reward_service_latency < 0:
        #     reward_service_latency = 0
        reward_service_latency = 1 / (path_cost + 1)

        # failure_score = self.current_batch[action][...]
        # reward_service_availability = 1 - failure_score

        reward_service_type = 1 \
            if service.get_service_type() == 3 and dest_machine.id == 0 \
            else 1 / (service.service_profile.e2e_latency - path_cost + 1)

        # REWARD = 0.5 * reward_service_latency + 0.5 * reward_service_type
        REWARD = 1 / ((1 - reward_service_latency) + (1 - reward_service_type) + 1e-1)

        # return dest_machine, action, torch.FloatTensor([REWARD])
        return dest_machine, action, REWARD

    def get_state(self, mec_net, service):
        # Construct valid (M, s_i) pairs.
        machine_service_pairs = []
        for machine in mec_net.machines:
            if self.can_satisfy_e2e_latency(mec_net, service, machine):
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
