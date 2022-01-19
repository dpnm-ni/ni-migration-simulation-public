import numpy as np
from abc import ABC, abstractmethod


class Algorithm(ABC):
    @abstractmethod
    def __call__(self, cluster, clock):
        pass


class RandomAlgorithm(Algorithm):
    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        candidates = []
        for service in waiting_services:
            for machine in machines:
                if machine.can_accommodate(service.service_profile):
                    candidates.append((machine, service))

        if len(candidates) == 0:
            return None, None
        else:
            rand_index = np.random.randint(0, len(candidates))
            return candidates[rand_index]


class FirstFitAlgorithm(Algorithm):
    # TODO: ensure to cover any failure case
    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        for service in waiting_services:
            for machine in machines:
                if machine.can_accommodate(service.service_profile):
                    return machine, service

        return None, None


# edge machine first. if not available, go to an available machine with least cost
# in ISP view (and current alg impl.), it is not necessarily to be an adjacent one (but graph property ensures it?)
class LeastCostAlgorithm(Algorithm):
    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        for service in waiting_services:
            # if edge machine is not in the current machine list (e.g. different #machine topo assumes), find alternate machine
            if service.edge_machine_id in [machine.id for machine in machines]:
                if machines[service.edge_machine_id].can_accommodate(service.service_profile):
                    edge_machine = machines[service.edge_machine_id]
                    return edge_machine, service
                else:
                    candidate_machine_ids = [machine.id for machine in machines if machine.id != service.edge_machine_id]
                    least_cost_machine_id = mec_net.get_least_cost_dest(service.edge_machine_id, candidate_machine_ids)
                    if machines[least_cost_machine_id].can_accommodate(service.service_profile):
                        print("[{}][{}] Service {} from Edge {} cannot run on its edge machine. go to Machine {}".format(
                            self.__class__.__name__, clock,
                            service.id, service.edge_machine_id, machines[least_cost_machine_id].id))
                        return machines[least_cost_machine_id], service

        return None, None
