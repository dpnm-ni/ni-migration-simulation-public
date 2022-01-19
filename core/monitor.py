class SLAMonitor:
    def __init__(self, env):
        self.env = env
        self.simulation = None
        self.mec_net = None
        self.accum_path_cost = 0

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(5)

        # while not self.simulation.is_finished():
        while len(self.mec_net.get_unfinished_services()) > 0:
            sum_path_cost = 0
            for service in self.mec_net.get_unfinished_services():
                path_cost = self.mec_net.get_path_cost(service.edge_machine_id, service.machine.id)
                # print("[{}][{}] RTT of Service {} is {} (src:{} <=> dst:{})"
                #       .format(self.__class__.__name__, self.env.now, service.id,
                #               path_cost * 2, service.edge_machine_id, service.machine.id))
                sum_path_cost += path_cost

            print("[{}][{}] average path cost is {}".format(
                self.__class__.__name__, self.env.now, sum_path_cost / len(self.mec_net.get_unfinished_services())))
            self.accum_path_cost += sum_path_cost

            # monitor performance of each service request/instance every 5s
            yield self.env.timeout(5)
