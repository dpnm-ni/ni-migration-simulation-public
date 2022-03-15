from base_logger import log

MONITOR_INTERVAL = 5
DISK_OVERUTIL_THRESHOLD = 0.8
NUM_SERVICE_TYPES = 4

class SLAMonitor:
    def __init__(self, env):
        self.env = env
        self.simulation = None
        self.mec_net = None

        self.accum_path_cost = 0
        self.hist_path_costs = []
        self.hist_types_path_costs = [[] for _ in range(NUM_SERVICE_TYPES)]

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    # FIXME:
    def monitor_service_path_cost(self):
        sum_path_cost = 0
        sum_types_path_cost = [0] * NUM_SERVICE_TYPES
        cnt_types = [0] * NUM_SERVICE_TYPES

        running_services = self.mec_net.get_unfinished_services()
        for service in running_services:
            service_path_cost = self.mec_net.get_path_cost(service.user_loc, service.machine.id)
            sum_path_cost += service_path_cost
            sum_types_path_cost[service.get_service_type()] += service_path_cost
            cnt_types[service.get_service_type()] += 1
        self.accum_path_cost += sum_path_cost

        if len(running_services) > 0:
            cur_avg_path_cost = sum_path_cost / len(running_services)
            self.hist_path_costs.append(cur_avg_path_cost)
            log.debug("[{}] average path cost for currently running services is {}".format(self.env.now, cur_avg_path_cost))

        for i, sum_type_path_cost, cnt_type in zip(range(NUM_SERVICE_TYPES), sum_types_path_cost, cnt_types):
            if cnt_type != 0:
                avg_type_path_cost = sum_type_path_cost / cnt_type
                self.hist_types_path_costs[i].append(avg_type_path_cost)
            else:
                self.hist_types_path_costs[i].append(0)

    def monitor_machine_resource_utilization(self):
        machines = self.mec_net.machines
        for machine in machines:
            if machine.destroyed is True:
                continue

            machine.mon_cpu_utilization = 1 - (machine.cpu / machine.cpu_capacity)
            machine.mon_memory_utilization = 1 - (machine.memory / machine.memory_capacity)
            machine.mon_disk_utilization = 1 - (machine.disk / machine.disk_capacity)
            if machine.mon_disk_utilization >= DISK_OVERUTIL_THRESHOLD:
                machine.mon_disk_overutil_cnt += 1
                if machine.mon_disk_overutil_cnt > 1:
                    log.info("[{}] Machine {}'s disk is under over-utilization {}".format(
                        self.env.now, machine.id, machine.mon_disk_overutil_cnt))
            # Initialize the counter if not consecutive.
            else:
                machine.mon_disk_overutil_cnt = 0

            machine.mon_cpu_util_hist.append(machine.mon_cpu_utilization)
            machine.mon_memory_util_hist.append(machine.mon_memory_utilization)
            machine.mon_disk_util_hist.append(machine.mon_disk_utilization)

    def run(self):
        yield self.env.timeout(MONITOR_INTERVAL)
        # Update the following values monitored at this tick.
        while not self.simulation.is_finished():
            # Accumulate path costs of running services and compute the average path cost (service latency).
            self.monitor_service_path_cost()

            # Update resource utilization of all machines and watch the occurrence of disk over-utilization.
            self.monitor_machine_resource_utilization()

            yield self.env.timeout(MONITOR_INTERVAL)
