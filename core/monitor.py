from base_logger import log

DISK_OVERUTIL_THRESHOLD = 0.8


class SLAMonitor:
    def __init__(self, env):
        self.env = env
        self.simulation = None
        self.mec_net = None

        self.accum_path_cost = 0
        self.cur_avg_path_cost = 0
        self.hist_path_costs = []

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(5)
        # yield self.env.timeout(1)

        while not self.simulation.is_finished():
            sum_path_cost = 0
            for service in self.mec_net.get_unfinished_services():
                path_cost = self.mec_net.get_path_cost(service.user_loc, service.machine.id)
                # log.debug("[{}] RTT of Service {} is {} (src:{} <=> dst:{})".format(
                #     self.env.now, service.id, path_cost * 2, service.edge_machine_id, service.machine.id))
                sum_path_cost += path_cost
            # !모니터링 틱 마다 현재 running 상태인 서비스들의 path cost(end latency)를 accum_path_cost에 누적시킴
            self.accum_path_cost += sum_path_cost

            if len(self.mec_net.get_unfinished_services()) > 0:
                self.cur_avg_path_cost = sum_path_cost / len(self.mec_net.get_unfinished_services())
                self.hist_path_costs.append(self.cur_avg_path_cost)
                log.debug("[{}] average path cost for currently running services is {}".format(
                    self.env.now, self.cur_avg_path_cost))

            machines = self.mec_net.machines
            for machine in machines:
                if machine.destroyed is True:
                    continue
                # FIXME:
                disk_utilization = 1 - (machine.disk / machine.disk_capacity)
                # disk_utilization = (machine.memory_capacity - machine.memory) / machine.disk_capacity
                machine.mon_disk_utilization = disk_utilization
                log.debug("[{}] Machine {}'s disk utilization: {}".format(self.env.now, machine.id, machine.mon_disk_utilization))

                if machine.mon_disk_utilization >= DISK_OVERUTIL_THRESHOLD:
                    machine.mon_disk_overutil_cnt += 1
                    if machine.mon_disk_overutil_cnt > 1:
                        log.debug("[{}] Machine {}'s disk is under over-utilization {}".format(
                            self.env.now, machine.id, machine.mon_disk_overutil_cnt))
                else:
                    machine.mon_disk_overutil_cnt = 0

            # monitor performance of each service request/instance every 5s
            yield self.env.timeout(5)
            # yield self.env.timeout(1)
