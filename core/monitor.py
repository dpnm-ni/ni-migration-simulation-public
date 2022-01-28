from base_logger import log


class SLAMonitor:
    def __init__(self, env):
        self.env = env
        self.simulation = None
        self.mec_net = None
        self.accum_path_cost = 0

        self.cur_path_costs = []
        self.cur_path_cost = 0

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        # yield self.env.timeout(5)
        yield self.env.timeout(1)

        while not self.simulation.is_finished():
        # while len(self.mec_net.get_unfinished_services()) > 0:
            sum_path_cost = 0
            for service in self.mec_net.get_unfinished_services():
                path_cost = self.mec_net.get_path_cost(service.edge_machine_id, service.machine.id)
                # log.debug("[{}] RTT of Service {} is {} (src:{} <=> dst:{})".format(
                #     self.env.now, service.id, path_cost * 2, service.edge_machine_id, service.machine.id))
                sum_path_cost += path_cost
            self.accum_path_cost += sum_path_cost

            if len(self.mec_net.get_unfinished_services()) > 0:
                self.cur_path_cost = sum_path_cost / len(self.mec_net.get_unfinished_services())
                self.cur_path_costs.append(self.cur_path_cost)
                log.debug("[{}] current average path cost is {}".format(self.env.now, self.cur_path_cost))

            # monitor performance of each service request/instance every 5s
            # yield self.env.timeout(5)
            yield self.env.timeout(1)
