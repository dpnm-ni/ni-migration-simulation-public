import numpy as np
import torch
from core.algorithm import Algorithm


class ActorCriticMigrationAlgorithm(Algorithm):
    def __init__(self, agent, num_epi, reward_giver=None):
        self.agent = agent
        self.agent.num_epi = num_epi
        self.reward_giver = reward_giver

    def __call__(self, mec_net, service=None):
        # Step 1: construct a placement map.
        placement_map = dict()
        running_services = mec_net.get_unfinished_services()
        for i in range(len(running_services)):
            service = running_services[i]

            # Filter out machines that are not accommodable and not SLA-compliant to the service.
            # -> note: feeding the resulted ragged array (state) into pytorch DNN is not supported,
            # so pad invalid machines too to keep the shape of the resulted array.
            candidate_machines = []
            for machine in mec_net.machines:
                # if self.can_satisfy_e2e_latency(mec_net, service, machine):
                #     if machine.can_accommodate(service.service_profile):
                #         candidate_machines.append(machine)
                candidate_machines.append(machine)

            placement_map[service] = candidate_machines

        # Step 2: create an input batch.
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
            # s_m_pairs = np.array(s_m_pairs).astype(float)
            batch.append(s_m_pairs)
        batch = np.array(batch).astype(float)

        # Step 3: create the current state.
        state = torch.from_numpy(batch).float()

        # Step 4: select an action set of [(s1, M), (s2, M), ...].
        selected_machines, probs = self.agent.get_action(state)

        # Step 5: for each service, check the selected machine (action) is valid.
        # If valid, service will be migrated to the selected machine
        # else, service stays in the current machine.
        destination_machines = [None] * len(running_services)
        for i, service in zip(range(len(running_services)), placement_map.keys()):
            assert running_services[i] == service

            machines = placement_map[service]
            new_machine = machines[selected_machines[i]]
            if not self.can_satisfy_e2e_latency(mec_net, service, new_machine) or not new_machine.can_accommodate(service.service_profile):
                destination_machines[i] = service.machine
            else:
                destination_machines[i] = new_machine

        # Step 6: compute reward?

        print("debug point")
