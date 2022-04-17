import numpy as np
import torch
from base_logger import log
from core.algorithm import Algorithm


class ActorCriticv2MigrationAlgorithm(Algorithm):
    def __init__(self, agent, num_epi, reward_giver):
        self.agent = agent
        self.agent.num_epi = num_epi

        self.reward_giver = reward_giver

    def make_batch(self, service, machines):
        batch = []
        for machine in machines:
            # feature_vector = [machine.cpu, machine.memory, machine.disk, machine.mon_disk_utilization] \
            #                  + [machine.machine_profile.edgeDC_id] \
            #                  + [service.service_profile.cpu, service.service_profile.memory, service.service_profile.disk] \
            #                  + [service.service_profile.e2e_latency] \
            #                  + [machine.mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)]
            """
            machine/service availability 관련 feature 제거 (DDPG 논문 Fig.2 참조)
            machine.mon_disk_utilization 포함 유무(위에 것)에 따라 (초기?) 학습 시간 2배 차이남 (sim_time: 1670 -> 1680)
            특정 edge/service의 provisioning delay 증가에 따른 것으로 추측되는데... 현재 result에는 average 처리해서 확연히 드러나지 않음
            """
            feature_vector = [machine.cpu, machine.memory, machine.disk] \
                             + [service.service_profile.cpu, service.service_profile.memory, service.service_profile.disk] \
                             + [service.service_profile.e2e_latency] \
                             + [machine.mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)]
            batch.append(feature_vector)
        return np.array(batch, dtype=np.float32)

    def __call__(self, mec_net, service):
        if service.migrating is True:
            print("service is migrating now")
            return None

        # Step 1: find available machines that ensure SLA constraints of the given service.
        candidate_machines = []
        for machine in mec_net.machines:
            # if service.migrating is False and \
            #         service.can_allow_availability() is True and \
            #         self.can_satisfy_e2e_latency(mec_net, service, machine) and \
            #         machine.can_accommodate(service.service_profile):
            # FIXME: DDPG 논문과의 비교 위해 availability 체크 해제 (해당 논문에서 고려 안함)
            if self.can_satisfy_e2e_latency(mec_net, service, machine) \
                    and machine.can_accommodate(service.service_profile):
                candidate_machines.append(machine)
        # When no machines are available for now.
        if len(candidate_machines) == 0:
            return None

        # Step 2: create an input batch for NN.
        batch = self.make_batch(service, candidate_machines)

        # Step 3: convert the batch into the corresponding state.
        state = torch.from_numpy(batch).float()

        # Step 4: select an action (selected machine as migration destination).
        dest_machine_index, prob_to_select = self.agent.get_action(state)

        # Step 5: migrate the service to the destination machine.
        src_machine = service.machine
        dest_machine = candidate_machines[dest_machine_index]
        if src_machine.id != dest_machine.id:
            service.live_migrate_service_instance(src_machine, dest_machine)
        else:
            log.debug("[{}] Service {} stays in the current M{}@E{}".format(
                service.env.now, service.id, src_machine, src_machine.machine_profile.edgeDC_id))
            # # FIXME: 자기자신으로의 migration도 action으로 취급 -> 이 경우 ensure reward == 0
            # service.live_migrate_service_instance(src_machine, src_machine)

        # Step 6: get the next state.
        batch = self.make_batch(service, candidate_machines)
        next_state = torch.from_numpy(batch).float()

        # Step 7: compute an instant reward for the migration action.
        latency_before = mec_net.get_path_cost(source_id=service.user_loc, dest_id=src_machine.id)
        latency_after = mec_net.get_path_cost(source_id=service.user_loc, dest_id=dest_machine.id)
        availability_before = 1 - src_machine.compute_failure_score(hist_window_size=5)
        availability_after = 1 - dest_machine.compute_failure_score(hist_window_size=5)

        reward = self.reward_giver(latency_before, latency_after,
                                   availability_before, availability_after)

        # Step 8: return the transition (s, a, r, s').
        # TODO: index or machine id? 만약 id가 action이라면 get_action 변경 필요.
        action = torch.tensor(dest_machine_index, dtype=torch.int64)
        # action = dest_machine.id
        return state, action, reward, next_state
