from base_logger import log

INJECTOR_INTERVAL = 5
DISK_FAULT_THRESHOLD = 2


class FaultInjector:
    def __init__(self, env):
        self.env = env
        self.simulation = None
        self.mec_net = None

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(INJECTOR_INTERVAL)
        while not self.simulation.is_finished():
            machines = self.mec_net.machines
            for machine in machines:
                # !일단 주사위 안던지고 count 임계치만 넘으면 서버 fault 처리
                if machine.mon_disk_overutil_cnt >= DISK_FAULT_THRESHOLD:
                    log.info("[{}] Machine {} at Edge {} has been failed. running instances will be interrupted...".format(
                        self.env.now, machine.id, machine.machine_profile.edgeDC_id))
                    # machine.destroy()

            yield self.env.timeout(INJECTOR_INTERVAL)
