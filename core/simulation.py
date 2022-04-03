from migration.ACv3.mig_controller import ActorCriticv3MigrationController


class Simulation:
    def __init__(self, env, mec_net, service_broker, scheduler, monitor, injector, controller):
        self.env = env
        self.mec_net = mec_net
        self.service_broker = service_broker
        self.scheduler = scheduler
        self.monitor = monitor
        self.injector = injector
        self.controller = controller

    def run(self):
        if self.service_broker is not None:
            self.service_broker.attach(self)
            self.env.process(self.service_broker.run())
        if self.scheduler is not None:
            self.scheduler.attach(self)
            self.env.process(self.scheduler.run())
        if self.monitor is not None:
            self.monitor.attach(self)
            self.env.process(self.monitor.run())
        if self.injector is not None:
            self.injector.attach(self)
            self.env.process(self.injector.run())
        if self.controller is not None:
            if isinstance(self.controller, list):
                for i in range(len(self.controller)):
                    self.controller[i].edgeDC_id = i
                    self.controller[i].attach(self)
                    self.env.process(self.controller[i].run())
            else:
                self.controller.attach(self)
                self.env.process(self.controller.run())

    def is_finished(self):
        return self.service_broker.destroyed \
               and len(self.mec_net.get_waiting_services()) == 0 \
               and len(self.mec_net.get_unfinished_services()) == 0
