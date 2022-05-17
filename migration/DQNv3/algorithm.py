import random
import numpy as np
import torch
from core.algorithm import Algorithm
from enum import IntFlag


class MIGRATION_STATUS(IntFlag):
    OK = 0
    ONGOING = 1
    VIOLATION_SLA_AVAILABILITY_MIGRATION = 2
    VIOLATION_SLA_AVAILABILITY_FAILURE = 4
    VIOLATION_SLA_LATENCY = 4
    NO_ACCOMMODABLE_MACHINES = 8
    SAME_SOURCE_DESTINATION = 16


class DQNv3MigrationAlgorithm(Algorithm):
    def __init__(self, agents, num_epi, reward_giver):
        assert isinstance(agents, list)
        self.agents = agents
        for i in range(len(self.agents)):
            agents[i].attach_algorithm(self)
            agents[i].num_epi = num_epi

        self.reward_giver = reward_giver

    def compute_edge_profile(self, mec_net, edgeDC_id):
        edge_machines = mec_net.get_edge_machines(edgeDC_id)

        sum_edge_cpu = 0
        for machine in edge_machines:
            sum_edge_cpu += machine.cpu
        avg_edge_cpu = sum_edge_cpu / len(edge_machines)

        sum_edge_mem = 0
        for machine in edge_machines:
            sum_edge_mem += machine.memory
        avg_edge_mem = sum_edge_mem / len(edge_machines)

        sum_edge_disk = 0
        for machine in edge_machines:
            sum_edge_disk += machine.disk
        avg_edge_disk = sum_edge_disk / len(edge_machines)

        sum_edge_disk_util = 0
        for machine in edge_machines:
            sum_edge_disk_util += machine.mon_disk_utilization
        avg_edge_disk_util = sum_edge_disk_util / len(edge_machines)

        return edge_machines[0], avg_edge_cpu, avg_edge_mem, avg_edge_disk, avg_edge_disk_util

    # As analogy to ACv2 who creates mapping features between each service to each machine,
    # for each service running in this DC, create its mapping features to each DC.
    # def make_batch(self, mec_net, service, edgeDC_profiles):
    #     batch = []
    #     for i in range(len(edgeDC_profiles)):
    #         sample_edge_machine, avg_edge_cpu, avg_edge_mem, avg_edge_disk, avg_edge_disk_util = edgeDC_profiles[i]
    #
    #         feature_vector = [avg_edge_cpu, avg_edge_mem, avg_edge_disk, avg_edge_disk_util] \
    #                          + [service.service_profile.cpu, service.service_profile.memory, service.service_profile.disk] \
    #                          + [service.service_profile.e2e_latency, service.service_profile.e2e_availability] \
    #                          + [mec_net.get_path_cost(source_id=service.user_loc, dest_id=sample_edge_machine.id)]
    #         batch.append(feature_vector)
    #
    #     return np.array(batch, dtype=np.float32)
    # Normalization ver.
    def make_batch(self, mec_net, service, edgeDC_profiles):
        batch = []
        for i in range(len(edgeDC_profiles)):
            sample_edge_machine, avg_edge_cpu, avg_edge_mem, avg_edge_disk, avg_edge_disk_util = edgeDC_profiles[i]

            feature_vector = [avg_edge_cpu, avg_edge_mem, avg_edge_disk, avg_edge_disk_util] \
                             + [service.service_profile.cpu, service.service_profile.memory, service.service_profile.disk] \
                             + [service.service_profile.e2e_latency, service.service_profile.e2e_availability] \
                             + [mec_net.get_path_cost(source_id=service.user_loc, dest_id=sample_edge_machine.id)]
            batch.append(feature_vector)

        batch = np.array(batch, dtype=np.float32)
        for col, i in zip(batch.T, range(batch.T.shape[0])):
            if i not in [4, 5, 6, 7, 8]:
                batch[:, i] = np.array(self.normalize(col, 0, 1))

        return batch

    def normalize(self, arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / (diff_arr + 1e-9)) + t_min
            norm_arr.append(temp)
        return norm_arr

    def get_accommodable_edge_machines(self, mec_net, service, edgeDC_id):
        edge_machines = mec_net.get_edge_machines(edgeDC_id)
        return [machine for machine in edge_machines if machine.can_accommodate(service.service_profile)]

    def __call__(self, mec_net, edgeDC_id):
        edge_running_services = mec_net.get_edge_unfinished_services(edgeDC_id)
        # TODO: migration 대상 서비스가 없을 시 skip (정책 없음?)
        if len(edge_running_services) == 0:
            return None

        # Step 0: do some preprocessings.
        sum_latency = 0
        for service in edge_running_services:
            sum_latency += mec_net.get_path_cost(service.user_loc, service.machine.id)
        avg_latency_before = sum_latency / len(edge_running_services)

        sum_availability = 0
        for service in edge_running_services:
            # Note: availability = 1 - failure_score(M)
            sum_availability += (1 - service.machine.compute_failure_score(hist_window_size=5))
        avg_availability_before = sum_availability / len(edge_running_services)

        # Step 1: compute the current profile of each edge statically.
        edgeDC_profiles = []
        for edgeDC_id in range(len(mec_net.edgeDCs)):
            edgeDC_profiles.append(self.compute_edge_profile(mec_net, edgeDC_id))

        # Step 2: create an input batch for each service.
        batches_per_service = []
        for service in edge_running_services:
            batch = self.make_batch(mec_net, service, edgeDC_profiles)
            batches_per_service.append(batch)

        # Step 3: convert the whole batches for all services (not per-service batch)
        # into the corresponding state (service-edge/DC placement map).
        state = torch.from_numpy(np.array(batches_per_service, dtype=np.float32)).float()

        # Step 4: select an action per service (selected machine as migration destination).
        # state.shape[0]: # of local services running in this DC.
        assert len(batches_per_service) == state.shape[0]
        dest_edge_ids = []
        for i in range(state.shape[0]):
            # Note that state[i] represents [(srv_i, e0), ... (srv_i, e15)] to compute fitness scores of those mapping.
            # This concept must be consistent in multi-agent DRL model regardless of algorithms.
            dest_edge_id = self.agents[edgeDC_id].get_action(state[i])
            dest_edge_ids.append(dest_edge_id)

        # Step 5: migrate the service to the destination machine.
        assert len(edge_running_services) == len(dest_edge_ids)
        service_migration_status = [MIGRATION_STATUS.OK] * len(edge_running_services)
        # FIXME:
        dest_edge_machines = [service.machine for service in edge_running_services]
        for i in range(len(edge_running_services)):
            service = edge_running_services[i]
            if service.migrating is True:
                service_migration_status[i] = MIGRATION_STATUS.ONGOING
                # print("[{}] ongoing".format(edgeDC_id))
                continue

            if service.can_allow_availability() is False:
                service_migration_status[i] = MIGRATION_STATUS.VIOLATION_SLA_AVAILABILITY_MIGRATION
                continue

            dest_edge_id = dest_edge_ids[i]
            candidate_dest_machines = self.get_accommodable_edge_machines(mec_net, service, dest_edge_id)
            if len(candidate_dest_machines) == 0:
                service_migration_status[i] = MIGRATION_STATUS.NO_ACCOMMODABLE_MACHINES
                continue

            # random.shuffle(candidate_dest_machines)
            src_machine = service.machine
            for j in range(len(candidate_dest_machines)):
                dest_machine = candidate_dest_machines[j]
                if self.can_satisfy_e2e_latency(mec_net, service, dest_machine):
                    if src_machine.id != dest_machine.id:
                        service_migration_status[i] = MIGRATION_STATUS.OK
                        dest_edge_machines[i] = dest_machine
                        service.live_migrate_service_instance(src_machine, dest_machine)
                        break
                    else:
                        # FIXME: not reachable?
                        service_migration_status[i] = MIGRATION_STATUS.SAME_SOURCE_DESTINATION
                if j == len(candidate_dest_machines) - 1:
                    service_migration_status[i] = MIGRATION_STATUS.VIOLATION_SLA_LATENCY
                    # FIXME:
                    # dest_edge_machines[i] = dest_machine
                    # service.live_migrate_service_instance(src_machine, dest_machine)

        # Step 6: get the next state.
        # batch = self.make_batch(mec_net, edge_running_services)
        # next_state = torch.from_numpy(batch).float()
        edgeDC_profiles_after = []
        for edgeDC_id in range(len(mec_net.edgeDCs)):
            edgeDC_profiles_after.append(self.compute_edge_profile(mec_net, edgeDC_id))
        batches_per_service_after = []
        for service in edge_running_services:
            batch = self.make_batch(mec_net, service, edgeDC_profiles_after)
            batches_per_service_after.append(batch)
        next_state = torch.from_numpy(np.array(batches_per_service_after, dtype=np.float32)).float()

        # Step 7: compute an instant reward for the migration action.
        assert len(edge_running_services) == len(service_migration_status) == len(dest_edge_machines)
        sum_latency = 0
        sum_availability = 0
        for i in range(len(edge_running_services)):
            service = edge_running_services[i]
            if service_migration_status[i] == MIGRATION_STATUS.OK or \
                    service_migration_status[i] == MIGRATION_STATUS.ONGOING or \
                    service_migration_status[i] == MIGRATION_STATUS.NO_ACCOMMODABLE_MACHINES or\
                    service_migration_status[i] == MIGRATION_STATUS.SAME_SOURCE_DESTINATION or \
                    service_migration_status[i] == MIGRATION_STATUS.VIOLATION_SLA_AVAILABILITY_MIGRATION:
                latency = mec_net.get_path_cost(service.user_loc, service.machine.id)
                failure_score = dest_edge_machines[i].compute_failure_score(hist_window_size=5)
            elif service_migration_status[i] == MIGRATION_STATUS.VIOLATION_SLA_AVAILABILITY_FAILURE:
                latency = mec_net.get_path_cost(service.user_loc, service.machine.id)
                # FIXME: 고장 가능성 높은 서버 피하는 행위 유도
                failure_score = 1
            elif service_migration_status[i] == MIGRATION_STATUS.VIOLATION_SLA_LATENCY:
                # FIXME:
                latency = 100
                # latency = mec_net.max_path_cost
                failure_score = dest_edge_machines[i].compute_failure_score(hist_window_size=5)
            else:
                latency = None
                failure_score = None

            sum_latency += latency
            sum_availability += (1 - failure_score)

        avg_latency_after = sum_latency / len(edge_running_services)
        avg_availability_after = sum_availability / len(edge_running_services)

        reward = self.reward_giver(avg_latency_before, avg_latency_after,
                                   avg_availability_before, avg_availability_after)

        # Step 8: return the transition (s, a, r, s').
        # TODO: index or machine id? 만약 id가 action이라면 get_action 변경 필요.
        action = torch.from_numpy(np.array(dest_edge_ids, dtype=np.int64))
        # action = dest_machine.id
        return state, action, reward, next_state
