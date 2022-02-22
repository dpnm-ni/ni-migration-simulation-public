import numpy as np
from abc import ABC, abstractmethod
from base_logger import log


class Algorithm(ABC):

    @abstractmethod
    def __call__(self, cluster, clock):
        pass

    def can_satisfy_e2e_latency(self, mec_net, service, machine):
        expected_rtt = 2 * mec_net.get_path_cost(service.user_loc, machine.id)
        if expected_rtt <= service.service_profile.e2e_latency:
            return True
        else:
            return False

# FIXME: random alg. becomes too slow with Edgenet. find another impl.
class RandomAlgorithm(Algorithm):

    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        candidates = []
        for service in waiting_services:
            for machine in machines:
                if self.can_satisfy_e2e_latency(mec_net, service, machine):
                    if machine.can_accommodate(service.service_profile):
                        candidates.append((machine, service))

        if len(candidates) == 0:
            return None, None
        else:
            rand_index = np.random.randint(0, len(candidates))
            return candidates[rand_index]


class FirstFitAlgorithm(Algorithm):

    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        for service in waiting_services:
            for machine in machines:
                if self.can_satisfy_e2e_latency(mec_net, service, machine):
                    if machine.can_accommodate(service.service_profile):
                        return machine, service

        return None, None


# TODO: try part can be included in except part?
class LeastCostAlgorithm(Algorithm):

    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        for service in waiting_services:
            # try to deploy a service in its closest edge DC
            try:
                machine_to_deploy = self.get_first_fit_edgeDC_machine(mec_net, service, service.user_loc, machines)
                if machine_to_deploy is not None:
                    return machine_to_deploy, service
                else:
                    # if no available machines in the closest edge DC
                    raise Exception

            # try next closest edge DC until available one found
            except (IndexError, Exception):
                least_cost_edgeDCs = mec_net.get_least_cost_edgeDCs(service.user_loc)
                # ensure edgeDCs are sorted according to their path costs from the service in ascending order
                for edgeDC_id in least_cost_edgeDCs:
                    machine_to_deploy = self.get_first_fit_edgeDC_machine(mec_net, service, edgeDC_id, machines)
                    if machine_to_deploy is not None:
                        log.debug(
                            "[{}] Service {} from Edge {} cannot be deployed in its edge DC. go to Machine {} at Edge {}"
                                .format(clock, service.id, service.user_loc, machine_to_deploy.id, edgeDC_id))
                        return machine_to_deploy, service

            # indication that scheduling is delayed
            log.debug("[{}] Service {} from Edge {} cannot be deployed in any edge DCs. skip it for now".format(
                clock, service.id, service.user_loc))

        return None, None

    def get_first_fit_edgeDC_machine(self, mec_net, service, edgeDC_id, machines):
        edgeDC_machines = [machine for machine in machines if
                           machine.machine_profile.edgeDC_id == edgeDC_id]
        for i in range(len(edgeDC_machines)):
            if self.can_satisfy_e2e_latency(mec_net, service, edgeDC_machines[i]):
                if edgeDC_machines[i].can_accommodate(service.service_profile):
                    # choose the firstfit machine
                    return edgeDC_machines[i]
        return None
