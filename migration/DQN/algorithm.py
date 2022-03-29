import numpy as np
import torch
from torch.distributions import Categorical
from core.algorithm import Algorithm
from core.injector import DISK_FAULT_THRESHOLD
from base_logger import log
from decimal import Decimal


class DQNMigrationAlgorithm(Algorithm):
    def __init__(self, agent, reward_giver=None):
        self.agent = agent
        self.reward_giver = reward_giver

        # Valid at the current state only. Should be initialized before the next state.
        self.current_machine_service_pairs = None
        self.current_batch = None

    def compute_migration_cost(self, mec_net, service, source_machine, dest_machine):
        # Migration cost is zero in the case of current-to-current.
        if source_machine.id == dest_machine.id:
            return [0, 0]
        # TODO: need to define a refined model for migration cost.
        else:
            # !임시로 정의: total mig time (VM transmission cost) + mig service downtime (larger if write-intensive app)
            # !MT: (service.memory + service.disk) * #hops
            # !DT: service.memory / service.duration (=write intensity 또는 service.machine.mem_utilization / #services 등...)
            MT = (service.memory + service.disk) * mec_net.get_path_length(
                source_id=source_machine.machine_profile.edgeDC_id, dest_id=dest_machine.id)
            DT = service.memory / round(Decimal(service.duration), 9)

            # !scale 다른데 normalization 필요? 가중치 필요?
            # return [MT + DT]
            return [MT, DT]

    # TODO: need to define a refined model for failure score prediction. DL?
    def compute_failure_score(self, machine):
        return [machine.mon_disk_overutil_cnt]

    # machine.cpu: remaining cpu cores
    def compute_resource_utilization(self, machine):
        return [1 - (machine.cpu/machine.cpu_capacity),
                1 - (machine.memory/machine.memory_capacity),
                1 - (machine.disk/machine.disk_capacity)]

    def make_input_batch(self, mec_net):
        batch = []
        for machine, service in self.current_machine_service_pairs:
            # # Convert a pair of (m_j, s_i) into the corresponding feature vector.
            # # We expect an output fitness score of (m_j, s_i) should be higher when the following values are lower:
            # # 1. path cost from s_i.user_loc (not s_i.machine) to m_j (-> service latency) [dim: 1]
            # # 2. migration cost from s_i.machine to m_j [dim: 2]
            # # 3. predicted failure score of m_j [dim: 1]
            # # 4. resource utilization of m_j (-> resource LB) [dim: 3]
            # # !migration cost 및 failure score 등 basic metric 아닌 reward에 쓰일 metric을 state로 취급하는게 이론적으로 맞는지?
            # # !기존 deployment 알고리즘에서 그런식으로 하긴했는데... 우리 문제에서 말이 안되는건 아니니 일단 시도해보고 성능이 좋은 방향으로
            # # !Cartpole로 치면 수레를 좌/우로 움직였을 때 예측되는 봉 각도를 reward로 쓴다는건데... 문제가 달라서 난센스이긴 한데 고민해볼 것
            # # !(+ reward 모델을 버티면 0 또는 넘어지면 -1과 같이 단순화하는게 가능?)
            # feature_vector = [machine.mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)] \
            #                  + self.compute_migration_cost(mec_net, service, service.machine, machine)\
            #                  + self.compute_failure_score(machine) \
            #                  + self.compute_resource_utilization(machine)

            feature_vector = [mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)] \
                             + [machine.cpu, machine.memory, machine.disk] \
                             + [service.service_profile.cpu, service.service_profile.memory, service.service_profile.disk, service.service_profile.duration]

            batch.append(feature_vector)
        return np.array(batch).astype(float)

        # !normalization 하는게 맞는지? Cartpole에서는 state인 (수레 위치, 수레 속도, 봉 각도, 봉 각속도)를 뉴럴넷에 그대로 넣음
        # https://www.google.com/search?q=reinforcement+learning+state+normalization&oq=reinforcement+learning+state+normalization&aqs=chrome..69i57.8960j0j7&sourceid=chrome&ie=UTF-8
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # for machine, service in valid_pairs:
        #     feature_vector = [machine.mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)] \
        #                      + self.compute_migration_cost(service, service.machine, machine)\
        #                      + self.compute_failure_score(machine) \
        #                      + self.compute_resource_utilization(machine)
        #     batch.append(np.array(feature_vector).astype(float))
        # return scaler.fit_transform(batch)

    # !기존 서버에서 기존 서버로의 placement (no migration) 결과도 포함되어 있음에 유의
    # !=> 해당 케이스는 reward를 다르게 계산해야될 것으로 보이나 일단은 동일하게 취급 후 추후 세분화 예정
    def __call__(self, mec_net, state):
        # Return as an action a machine that maximizes Q(state, action),
        # where the state is batched feature vectors of (M, s_i) pairs.
        action = self.agent.get_action(state)

        # FIXME: use reward_giver.
        # Use the path cost of the selected (m_new, s_i) pair as a immediate reward.
        # The path cost is 0 if the machine is in the nearest edge DC to s_i.user_loc.
        # !엄밀하게는 해당 서비스의 duration까지 고려해서 계산해야됨 (expected accumulated path cost by this placement)
        # !일단은 테스팅 목적으로 단순하게 계산함 (duration 보장 안되는 고장 상황에서는 성능상 이게 나을수도?)
        path_cost = self.current_batch[action][0]
        # FIXME: define a refined reward function.
        reward_path_cost = np.log(mec_net.max_path_cost - path_cost + 1e-7)
        REWARD = reward_path_cost

        dest_machine = self.current_machine_service_pairs[action][0]
        return dest_machine, action, torch.FloatTensor([REWARD])

    def get_state(self, mec_net, service):
        # self.initialize_state_configs()

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
            state = torch.from_numpy(self.current_batch).float()
            return state
        # When no machines are available to the service for now.
        else:
            return None

    def initialize_state_configs(self):
        self.current_machine_service_pairs = None
        self.current_batch = None
