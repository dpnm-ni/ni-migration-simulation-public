from core.service import Service


# 개념적으로 모든 service 요청이 통과하는 proxy
class Broker:
    def __init__(self, env, service_profiles):
        self.env = env
        self.service_profiles = service_profiles

        self.simulation = None
        self.mec_net = None
        self.destroyed = False

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        for service_profile in self.service_profiles:
            assert service_profile.submit_time >= self.env.now

            # delay reception of the service request until its specified submit time
            service_submission_delay = service_profile.submit_time - self.env.now
            yield self.env.timeout(service_submission_delay)

            service = Service(self.env, service_profile)
            service.queued_timestamp = self.env.now
            print("[{}][{}] a request for Service {} arrived".format(self.__class__.__name__, service.queued_timestamp, service.id))
            self.mec_net.add_service(service)

        self.destroyed = True
