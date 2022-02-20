from base_logger import log


class Scheduler:
    def __init__(self, env, deployment_algorithm):
        self.env = env
        self.deployment_algorithm = deployment_algorithm

        self.simulation = None
        self.mec_net = None
        self.destroyed = False
        self.valid_pairs = []

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        while not self.simulation.is_finished():
            self.make_decision()

            # (1) imply a request just arrived needs to wait for at least 1s to be handled and
            # (2) wait 1s when decision failed
            # loop occurs if commented
            yield self.env.timeout(1)

        self.destroyed = True

    def make_decision(self):
        while True:
            log.debug("[{}] Pending service requests: {}".format(self.env.now, self.mec_net.get_waiting_services()))
            machine, service = self.deployment_algorithm(self.mec_net, self.env.now)
            if machine is None or service is None:
                break
            else:
                log.debug("[{}] Service {} from Edge {} started on Machine {} at Edge {} (duration: {})".format(
                    self.env.now, service.id, service.user_loc, machine.id, machine.machine_profile.edgeDC_id, service.duration))
                service.start_service_instance(machine)
                log.debug("[{}] Machine state after: {}".format(self.env.now, machine.get_state()))
                self.valid_pairs.append((machine.id, service.id))
