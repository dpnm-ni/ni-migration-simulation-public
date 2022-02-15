import numpy as np
from abc import ABC, abstractmethod
from base_logger import log


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
    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        for service in waiting_services:
            for machine in machines:
                if machine.can_accommodate(service.service_profile):
                    return machine, service

        return None, None


# go to edge machine first. if not available, go to an alternative machine with least cost
# in ISP view (and current alg impl.), it is not necessarily to be an adjacent one (but graph property ensures it?)
class LeastCostAlgorithm(Algorithm):
    def __call__(self, mec_net, clock):
        waiting_services = mec_net.get_waiting_services()
        machines = mec_net.machines
        for service in waiting_services:
            try:
                available = machines[service.edge_machine_id].can_accommodate(service.service_profile)
                if available == 0:
                    raise Exception
                else:
                    edge_machine = machines[service.edge_machine_id]
                    return edge_machine, service
            except (IndexError, Exception):
                candidate_machine_ids = [machine.id for machine in machines if machine.id != service.edge_machine_id]
                while True:
                    least_cost_machine_ids = mec_net.get_least_cost_dest_ids(service.edge_machine_id, candidate_machine_ids)
                    if least_cost_machine_ids is None:
                        break
                    for _id in least_cost_machine_ids:
                        if machines[_id].can_accommodate(service.service_profile):
                            alternative_machine = machines[_id]
                            log.debug(
                                "[{}] Service {} from Edge {} cannot run on its edge machine. go to Machine {}".format(
                                    clock, service.id, service.edge_machine_id, alternative_machine.id))
                            return alternative_machine, service

                        # exclude the not available machine
                        candidate_machine_ids.remove(_id)

        return None, None
