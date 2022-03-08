import simpy
from base_logger import log


class Service:
    def __init__(self, env, service_profile):
        self.env = env

        self.service_profile = service_profile
        self.id = self.service_profile.service_id
        self.submit_time = self.service_profile.submit_time
        self.cpu = self.service_profile.cpu
        self.memory = self.service_profile.memory
        self.disk = self.service_profile.disk
        self.duration = self.service_profile.duration
        # Should be updated when user movement event.
        self.user_loc = self.service_profile.user_loc

        # Machine where the instance of this service is actually deployed (by the deployment algorithm configured).
        self.machine = None
        self.work_event = None

        self.started = False
        self.finished = False
        self.queued_timestamp = None
        self.started_timestamp = None
        self.finished_timestamp = None

    def start_service_instance(self, machine):
        self.started = True
        self.started_timestamp = self.env.now

        self.machine = machine
        self.machine.run_service_instance(self)
        self.work_event = self.env.process(self.do_work())

    def live_migrate_service_instance(self, src_machine, dest_machine):
        assert self.started is True and self.finished is False
        assert self.machine == src_machine

        # Here start the live migration operation and generate related costs (total mig time + service downtime + @).
        # Note that SimPy does not wake up the interrupt handler synchronously.
        # So you expect the internal logic of the handler is executed after the end of this function.
        self.work_event.interrupt(cause=1)

        # In general, VM live migration in real world does Step2 first then Step1 at the end of image transmission.
        # Step1: stop the service instance running in src_machine.
        src_machine.stop_service_instance(self)
        self.machine = None

        # Step2: start the service instance at dest_machine.
        # Due to the asynchronous behavior of the interrupt handler, compute the remaining duration here.
        elapsed_time = self.env.now - self.started_timestamp
        self.duration = self.duration - elapsed_time if elapsed_time <= self.duration else 0
        self.start_service_instance(dest_machine)

    def do_work(self):
        try:
            yield self.env.timeout(self.duration)
            self.finished = True
            self.finished_timestamp = self.env.now
            log.info("[{}] Service {} finished".format(self.finished_timestamp, self.id))

            self.machine.stop_service_instance(self)
            log.debug("[{}] Machine state after: {}".format(self.env.now, self.machine.get_state()))

        # !서비스 중단 상황 (서버 fault 등). triggered by FaultInjector
        except simpy.Interrupt as interrupt:
            # Caused by machine (server) failure.
            if interrupt.cause == 0:
                # TODO: due to the asynchronous behavior of the interrupt handler, compute the remaining duration at the caller.
                elapsed_time = self.env.now - self.started_timestamp
                self.duration = self.duration - elapsed_time if elapsed_time <= self.duration else 0

                self.machine = None
                # Invalidating this flag ensures that this service gets rescheduled by Scheduler.
                self.started = False
                self.started_timestamp = None
                self.queued_timestamp = self.env.now
                log.info("[{}] Service {} (duration: {}) interrupted. go back to waiting queue".format(
                    self.env.now, self.id, self.duration))

            # Caused by live migration.
            elif interrupt.cause == 1:
                # elapsed_time = self.env.now - self.started_timestamp
                # self.duration = self.duration - elapsed_time if elapsed_time <= self.duration else 0

                log.debug("[{}] live migration penalty".format(self.env.now))

    # TODO: refactor the entire codes to be coherent in handling instance properties
    def is_started(self):
        return self.started

    def is_finished(self):
        return self.finished

    def __repr__(self):
        return str(self.id)
        # return "{}: [cpu: {}, memory: {}, disk: {}]".format(self.id, self.cpu, self.memory, self.disk)


# Intended to include only static information when the service request is created (e.g., from csv).
# It would be better to store runtime information (e.g., user mobility) in Serivce class itself.
class ServiceProfile:
    def __init__(self, service_id, submit_time, cpu, memory, disk, duration, user_loc, e2e_latency):
        self.service_id = service_id
        self.submit_time = submit_time
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.duration = duration

        # self.edge_machine_id = edge_machine_id
        self.user_loc = user_loc

        # A service's constraint for end-to-end latency.
        self.e2e_latency = e2e_latency
