from base_logger import log


class Service:
    global_index = -1

    def __init__(self, env, service_profile):
        self.env = env

        self.service_profile = service_profile
        # self.id = self.get_service_id()
        self.id = self.service_profile.service_id
        self.submit_time = self.service_profile.submit_time
        self.cpu = self.service_profile.cpu
        self.memory = self.service_profile.memory
        self.disk = self.service_profile.disk
        self.duration = self.service_profile.duration

        self.edge_machine_id = self.service_profile.edge_machine_id
        # machine where the instance of this service is actually deployed (by the deployment algorithm configured)
        self.machine = None
        # FIXME: is this used?
        self.work_event = None

        self.started = False
        self.finished = False
        self.queued_timestamp = None
        self.started_timestamp = None
        self.finished_timestamp = None

    # get service request number
    @classmethod
    def get_service_id(cls):
        cls.global_index += 1
        return cls.global_index

    def start_service_instance(self, machine):
        self.started = True
        self.started_timestamp = self.env.now

        self.machine = machine
        self.machine.run_service_instance(self)
        self.work_event = self.env.process(self.do_work())

    def do_work(self):
        yield self.env.timeout(self.duration)
        self.finished = True
        self.finished_timestamp = self.env.now
        log.debug("[{}] Service {} finished".format(self.finished_timestamp, self.id))

        self.machine.stop_service_instance(self)
        log.debug("[{}] Machine state after: {}".format(self.env.now, self.machine.get_state()))

    # TODO: refactor the entire codes to be coherent in handling instance properties
    def is_started(self):
        return self.started

    def is_finished(self):
        return self.finished

    def __repr__(self):
        return str(self.id)
        # return "{}: [cpu: {}, memory: {}, disk: {}]".format(self.id, self.cpu, self.memory, self.disk)


class ServiceProfile:
    def __init__(self, service_id, submit_time, cpu, memory, disk, duration, edge_machine_id):
        self.service_id = service_id
        self.submit_time = submit_time
        self.cpu = cpu
        self.memory = memory
        self.disk = disk
        self.duration = duration
        self.edge_machine_id = edge_machine_id
