import numpy as np
from abc import ABC, abstractmethod
from base_logger import log


# Base deployment algorithms.
class Algorithm(ABC):
    @abstractmethod
    def __call__(self, mec_net, service):
        pass

    def can_satisfy_e2e_latency(self, mec_net, service, machine):
        # expected_rtt = 2 * mec_net.get_path_cost(service.user_loc, machine.id)
        expected_rtt = mec_net.get_path_cost(service.user_loc, machine.id)
        return expected_rtt <= service.service_profile.e2e_latency


# FIXME: random alg. becomes too slow with Edgenet. find another impl.
class RandomAlgorithm(Algorithm):
    # def __call__(self, mec_net, service):
    #     waiting_services = mec_net.get_waiting_services()
    #     machines = mec_net.machines
    #     candidates = []
    #     for service in waiting_services:
    #         for machine in machines:
    #             # TODO: duplicate code?
    #             if self.can_satisfy_e2e_latency(mec_net, service, machine):
    #                 if machine.can_accommodate(service.service_profile):
    #                     candidates.append((machine, service))
    #     if len(candidates) == 0:
    #         return None, None
    #     else:
    #         rand_index = np.random.randint(0, len(candidates))
    #         return candidates[rand_index]
    def __call__(self, mec_net, service):
        candidate_machines = []
        machines = mec_net.machines
        for machine in machines:
            # TODO: duplicate code?
            if self.can_satisfy_e2e_latency(mec_net, service, machine):
                if machine.can_accommodate(service.service_profile):
                    candidate_machines.append(machine)
        if len(candidate_machines) == 0:
            return None
        else:
            rand_index = np.random.randint(0, len(candidate_machines))
            return candidate_machines[rand_index]


class FirstFitAlgorithm(Algorithm):
    # def __call__(self, mec_net, clock):
    #     waiting_services = mec_net.get_waiting_services()
    #     machines = mec_net.machines
    #     for service in waiting_services:
    #         for machine in machines:
    #             if self.can_satisfy_e2e_latency(mec_net, service, machine):
    #                 if machine.can_accommodate(service.service_profile):
    #                     return machine, service
    #     return None, None
    def __call__(self, mec_net, service):
        machines = mec_net.machines
        for machine in machines:
            if self.can_satisfy_e2e_latency(mec_net, service, machine):
                if machine.can_accommodate(service.service_profile):
                    return machine
        return None


# TODO: try part can be included in except part?
class LeastCostAlgorithm(Algorithm):
    # def __call__(self, mec_net, clock):
    #     waiting_services = mec_net.get_waiting_services()
    #     machines = mec_net.machines
    #     for service in waiting_services:
    #         # Try to deploy a service in its closest/nearest edge DC.
    #         try:
    #             # Note that service.user_loc itself maps its nearest edge DC (id), so their path cost is 0.
    #             machine_to_deploy = self.get_first_fit_edgeDC_machine(mec_net, service, service.user_loc, machines)
    #             if machine_to_deploy is not None:
    #                 return machine_to_deploy, service
    #             else:
    #                 # If no available machines in the closest edge DC.
    #                 raise Exception
    #
    #         # Try next closest edge DC until available one found.
    #         except (IndexError, Exception):
    #             least_cost_edgeDCs = mec_net.get_least_cost_edgeDCs(service.user_loc)
    #             # Ensure the resulted edgeDCs are sorted according to their path costs in ascending order.
    #             for edgeDC_id in least_cost_edgeDCs:
    #                 machine_to_deploy = self.get_first_fit_edgeDC_machine(mec_net, service, edgeDC_id, machines)
    #                 if machine_to_deploy is not None:
    #                     log.debug(
    #                         "[{}] Service {} from Edge {} cannot be deployed in its edge DC. go to Machine {} at Edge {}".format(
    #                             clock, service.id, service.user_loc, machine_to_deploy.id, edgeDC_id))
    #                     return machine_to_deploy, service
    #
    #         # Indication that this scheduling/deployment decision is delayed.
    #         log.debug("[{}] Service {} from Edge {} cannot be deployed in any edge DCs. skip it for now".format(
    #             clock, service.id, service.user_loc))
    #     return None, None
    def __call__(self, mec_net, service):
        machines = mec_net.machines
        # Try to deploy a service in its closest/nearest edge DC.
        try:
            # Note that service.user_loc itself maps its nearest edge DC (id), so their path cost is 0.
            machine_to_deploy = self.get_first_fit_edgeDC_machine(mec_net, service, service.user_loc, machines)
            if machine_to_deploy is not None:
                return machine_to_deploy
            else:
                # If no available machines in the closest edge DC.
                raise Exception

        # Try next closest edge DC until available one found.
        except (IndexError, Exception):
            least_cost_edgeDCs = mec_net.get_least_cost_edgeDCs(service.user_loc)
            # Ensure the resulted edgeDCs are sorted according to their path costs in ascending order.
            for edgeDC_id in least_cost_edgeDCs:
                machine_to_deploy = self.get_first_fit_edgeDC_machine(mec_net, service, edgeDC_id, machines)
                if machine_to_deploy is not None:
                    log.debug(
                        "[{}] Service {} from Edge {} cannot be deployed in its edge DC. go to Machine {} at Edge {}".format(
                            service.env.now, service.id, service.user_loc, machine_to_deploy.id, edgeDC_id))
                    return machine_to_deploy

        # Indication that this scheduling/deployment decision is delayed.
        log.debug("[{}] Service {} from Edge {} cannot be deployed in any edge DCs. skip it for now".format(
            service.env.now, service.id, service.user_loc))
        return None

    def get_first_fit_edgeDC_machine(self, mec_net, service, edgeDC_id, machines):
        edgeDC_machines = [machine for machine in machines
                           if machine.machine_profile.edgeDC_id == edgeDC_id]
        for i in range(len(edgeDC_machines)):
            if self.can_satisfy_e2e_latency(mec_net, service, edgeDC_machines[i]):
                if edgeDC_machines[i].can_accommodate(service.service_profile):
                    # Choose the firstfit machine in the given edge DC.
                    return edgeDC_machines[i]
        return None
