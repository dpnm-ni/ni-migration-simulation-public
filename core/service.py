import simpy
from base_logger import log

# Note: migration 횟수를 결정 짓는데 영향을 미치지만 0.1로 설정하더라도 migration 횟수가 10배 늘어나지 않는 이유는
# mig_interval 값이 dominate하기 때문. 만일 interval을 매우 줄인다면 해당 값도 횟수에 영향을 미칠 것으로 보임
SERVICE_DOWNTIME_CONSTANT = 1


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

        self.num_interruptions_by_fault = 0
        self.num_interruptions_by_migration = 0
        self.num_migrations_to_cloud = 0
        self.num_migrations_to_edge = 0

        # FIXME: refine the locking mechanism to represent migration performance depending on services
        # Lock should be taken when migration is ongoing and released after migration is done.
        self.migrating = False

        # Live migration service downtime related.
        self.hist_service_downtimes = []
        self.accum_service_downtime = 0

    def start_service_instance(self, machine):
        self.started = True
        self.started_timestamp = self.env.now

        self.machine = machine
        self.machine.run_service_instance(self)
        self.work_event = self.env.process(self.do_work())

    # Same as start_service_instance but maintain the instance's current session info (e.g., start_time).
    def start_service_instance_after_migration(self, machine):
        self.started = True
        # self.started_timestamp = self.env.now

        self.machine = machine
        self.machine.run_service_instance(self)
        self.work_event = self.env.process(self.do_work())

    def live_migrate_service_instance(self, src_machine, dest_machine):
        assert self.started is True and self.finished is False
        assert self.machine == src_machine

        # Take the migration lock.
        self.migrating = True

        # Here start the live migration operation and generate related costs (total mig time + service downtime + @).
        # Note that SimPy does not wake up the interrupt handler synchronously.
        # So you expect the internal logic of the interrupt handler is executed after the end of this function.
        self.work_event.interrupt(cause=1)

        # In general, VM live migration in real world does Step2 first then Step1 at the end of image transmission.
        # Step1: stop the service instance running in src_machine.
        src_machine.stop_service_instance(self)
        self.machine = None

        # Step2: start the service instance at dest_machine.
        # Note that the call for the service interrupt is asynchronous, so update the remaining duration here.
        elapsed_time = self.env.now - self.started_timestamp
        self.duration = self.duration - elapsed_time if elapsed_time <= self.duration else 0
        self.start_service_instance_after_migration(dest_machine)

        if dest_machine.machine_profile.edgeDC_id == 0:
            self.num_migrations_to_cloud += 1
        else:
            self.num_migrations_to_edge += 1

    def do_work(self):
        try:
            # Release the migration lock.
            self.migrating = False

            yield self.env.timeout(self.duration)
            self.finished = True
            self.finished_timestamp = self.env.now
            log.debug("[{}] Service {} finished".format(self.finished_timestamp, self.id))

            self.machine.stop_service_instance(self)
            log.debug("[{}] Machine state after: {}".format(self.env.now, self.machine.get_state()))

        # !서비스 중단 상황 (서버 fault 등). triggered by FaultInjector
        except simpy.Interrupt as interrupt:
            # Caused by machine (server) failure.
            if interrupt.cause == 0:
                self.machine = None
                # Invalidating this flag ensures that this service gets rescheduled by Scheduler.
                self.started = False
                self.started_timestamp = None
                self.queued_timestamp = self.env.now
                log.info("[{}] Service {} (duration: {}) interrupted. go back to waiting queue".format(
                    self.env.now, self.id, self.duration))

                self.num_interruptions_by_fault += 1

            # Caused by live migration.
            elif interrupt.cause == 1:
                # TODO:
                log.debug("[{}] live migration penalty".format(self.env.now))

                service_downtime = self.compute_service_downtime()
                self.hist_service_downtimes.append(service_downtime)
                self.accum_service_downtime += service_downtime

                self.num_interruptions_by_migration += 1

    def compute_service_downtime(self):
        memory_intensity = float(self.service_profile.memory) / self.service_profile.duration
        service_downtime = memory_intensity * SERVICE_DOWNTIME_CONSTANT
        return service_downtime

    def can_allow_availability(self):
        service_downtime = self.compute_service_downtime()
        if self.service_profile.e2e_availability == 0:
            return True
        else:
            availability = (1 - ((self.accum_service_downtime + service_downtime) / self.duration)) * 100
            if availability >= self.service_profile.e2e_availability:
                return True
            return False

    def is_started(self):
        return self.started

    def is_finished(self):
        return self.finished

    # FIXME:
    def get_service_type(self):
        if self.service_profile.e2e_latency == 5:
            return 0
        elif self.service_profile.e2e_latency == 10:
            return 1
        elif self.service_profile.e2e_latency == 50:
            return 2
        else:
            return 3

    def __repr__(self):
        return str(self.id)
        # return "{}: [cpu: {}, memory: {}, disk: {}]".format(self.id, self.cpu, self.memory, self.disk)


# Intended to include only static information when the service request is created (e.g., from csv).
# It would be better to store runtime information (e.g., user mobility) in Serivce class itself.
class ServiceProfile:
    def __init__(self, service_id, submit_time, cpu, memory, disk, duration, user_loc, e2e_latency, e2e_availability):
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
        self.e2e_availability = e2e_availability
