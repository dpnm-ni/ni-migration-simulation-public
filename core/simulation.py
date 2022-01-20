from core.monitor import SLAMonitor


class Simulation:
    def __init__(self, env, mec_net, service_broker, scheduler, monitor):
        self.env = env
        self.mec_net = mec_net
        self.service_broker = service_broker
        self.scheduler = scheduler
        self.monitor = monitor

        self.service_broker.attach(self)
        self.scheduler.attach(self)
        self.monitor.attach(self)

    def run(self):
        # process to submit service requests at their specified submit_time, instead of individual users
        self.env.process(self.service_broker.run())

        self.env.process(self.scheduler.run())

        self.env.process(self.monitor.run())

    # FIXME: current condition only ensures the service provisioning, not all the services' end.
    def is_finished(self):
        return self.service_broker.destroyed \
               and len(self.mec_net.get_waiting_services()) == 0 \
               and len(self.mec_net.get_unfinished_services()) == 0
