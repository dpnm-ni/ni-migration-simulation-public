from base_logger import log

DISK_FAULT_THRESHOLD = 9


class FaultInjector:
    def __init__(self, env):
        self.env = env
        self.simulation = None
        self.mec_net = None

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(5)

        while not self.simulation.is_finished():
            machines = self.mec_net.machines
            for machine in machines:
                # !일단 주사위 안던지고 count 임계치만 넘으면 서버 fault 처리
                if machine.mon_disk_overutil_cnt >= DISK_FAULT_THRESHOLD:
                    log.debug("[{}] Machine {} has been failed. running instances will be interrupted...".format(
                        self.env.now, machine.id))
                    machine.destroy()
                    machine.mon_disk_overutil_cnt = 0

            yield self.env.timeout(5)
