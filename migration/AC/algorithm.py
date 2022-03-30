import numpy as np
import torch
from base_logger import log
from core.algorithm import Algorithm


class ActorCriticMigrationAlgorithm(Algorithm):
    def __init__(self, agent, num_epi, reward_giver=None):
        self.agent = agent
        self.agent.num_epi = num_epi
        self.reward_giver = reward_giver

    def make_batch(self, placement_map, mec_net):
        batch = []
        for service in placement_map.keys():
            s_m_pairs = []
            machines = placement_map[service]
            for machine in machines:
                # TODO: duration should be included?
                s_m_pair = [mec_net.get_path_cost(source_id=service.user_loc, dest_id=machine.id)] \
                           + [machine.cpu, machine.memory, machine.disk] \
                           + [machine.mon_disk_utilization] \
                           + [service.service_profile.cpu, service.service_profile.memory, service.service_profile.disk,
                              service.service_profile.duration, service.service_profile.e2e_latency]
                s_m_pairs.append(s_m_pair)
            batch.append(s_m_pairs)
        batch = np.array(batch, dtype=np.float32)

        return batch

    def __call__(self, mec_net, service=None):
        # Step 1: do preprocessing on service placement info.
        # Step 1-1: construct a placement map.
        placement_map = dict()
        running_services = mec_net.get_unfinished_services()
        # FIXME:
        if len(running_services) == 0:
            return None
        for i in range(len(running_services)):
            service = running_services[i]
            # (old) filter out machines that are not accommodable and not SLA-compliant to the service.
            # (new) feeding the resulted ragged array (state) into pytorch DNN is not supported now,
            # so invalid machines are also added to keep the shape as square.
            candidate_machines = []
            for machine in mec_net.machines:
                # if self.can_satisfy_e2e_latency(mec_net, service, machine):
                #     if machine.can_accommodate(service.service_profile):
                #         candidate_machines.append(machine)
                candidate_machines.append(machine)
            placement_map[service] = candidate_machines

        # Step 1-2: compute the average service latency before migration.
        sum_latency = 0
        for service in placement_map.keys():
            sum_latency += mec_net.get_path_cost(service.user_loc, service.machine.id)
        avg_latency_before = sum_latency / len(placement_map.keys())

        # Step 1-3: compute the average failure score of migration source machines.
        sum_machine_failure_score = 0
        for service in placement_map.keys():
            sum_machine_failure_score += service.machine.compute_failure_score(hist_window_size=5)
        avg_machine_failure_score_before = sum_machine_failure_score / len(placement_map.keys())

        # Step 2: create an input batch for DNN.
        batch = self.make_batch(placement_map, mec_net)

        # Step 3: convert the batch into the corresponding current state .
        state = torch.from_numpy(batch).float()

        # Step 4: select an action set of [(s1, M), (s2, M), ...].
        selected_machines, probs = self.agent.get_action(state)

        # Step 5: adjust the action set as:
        # for each service, check the selected machine (action) is valid for migration.
        # If valid, service will be migrated to the selected machine
        # else, service stays in the current machine.
        destination_machines = [None] * len(running_services)
        for i, service in zip(range(len(running_services)), placement_map.keys()):
            assert running_services[i] == service
            machines = placement_map[service]
            src_machine = service.machine
            dest_machine = machines[selected_machines[i]]
            if service.migrating is False and \
                    service.can_allow_availability() is True and \
                    self.can_satisfy_e2e_latency(mec_net, service, dest_machine) and \
                    dest_machine.can_accommodate(service.service_profile):
                destination_machines[i] = dest_machine
                service.live_migrate_service_instance(src_machine, destination_machines[i])
            else:
                destination_machines[i] = src_machine
                # service.live_migrate_service_instance(src_machine, destination_machines[i])
                # FIXME:
                # log.debug("[{}] Service {} stays in the current M{}@E{}".format(
                #     self.agent.env.now, service.id, src_machine, src_machine.machine_profile.edgeDC_id))
                log.debug("Service {} stays in the current M{}@E{}".format(
                    service.id, src_machine, src_machine.machine_profile.edgeDC_id))

        # Step 7: get the next state.
        batch = self.make_batch(placement_map, mec_net)
        next_state = torch.from_numpy(batch).float()

        # Step 8: get the instant reward.
        # Step 8-1: compute the average service latency after migration.
        sum_latency = 0
        for service in placement_map.keys():
            sum_latency += mec_net.get_path_cost(service.user_loc, service.machine.id)
        avg_latency_after = sum_latency / len(placement_map.keys())

        # Step 8-2: compute the average failure score of migration destination machines.
        sum_machine_failure_score = 0
        for machine in destination_machines:
            sum_machine_failure_score += machine.compute_failure_score(hist_window_size=5)
        avg_machine_failure_score_after = sum_machine_failure_score / len(destination_machines)

        # Step 8-3: compute performance benefits after migration.
        if avg_latency_before == 0:
            L_gains = 0
        else:
            L_gains = (avg_latency_before - avg_latency_after) / avg_latency_before
        if avg_machine_failure_score_before == 0:
            A_gains = 0
        else:
            A_gains = (avg_machine_failure_score_before - avg_machine_failure_score_after) / avg_machine_failure_score_before
        reward = L_gains + A_gains

        # Step 9: return the transition (s, a, r, s').
        # return state, selected_machines, reward, next_state
        action = torch.from_numpy(np.array([machine.id for machine in destination_machines], dtype=np.int64)).view([-1, 1, 1])
        return state, action, reward, next_state
