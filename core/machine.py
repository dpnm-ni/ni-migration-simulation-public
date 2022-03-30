import numpy as np
from decimal import Decimal


# machine type: edge or core
class Machine:
    global_index = -1

    def __init__(self, machine_profile):
        self.id = Machine.get_global_id()
        self.mec_net = None

        self.machine_profile = machine_profile
        self.cpu_capacity = self.machine_profile.cpu_capacity
        self.memory_capacity = self.machine_profile.memory_capacity
        self.disk_capacity = self.machine_profile.disk_capacity
        self.cpu = self.machine_profile.cpu
        self.memory = self.machine_profile.memory
        self.disk = self.machine_profile.disk

        # periodically updated by Monitor
        self.mon_cpu_utilization = 0
        self.mon_cpu_util_hist = []
        self.mon_memory_utilization = 0
        self.mon_memory_util_hist = []
        self.mon_disk_utilization = 0
        self.mon_disk_util_hist = []
        self.mon_disk_overutil_cnt = 0

        self.running_service_instances = []
        self.destroyed = False

    @classmethod
    def get_global_id(cls):
        cls.global_index += 1
        return cls.global_index

    def attach(self, mec_net):
        self.mec_net = mec_net

    def run_service_instance(self, service):
        self.cpu -= service.cpu
        assert self.cpu >= 0
        self.memory -= service.memory
        assert self.memory >= 0
        self.disk -= service.disk
        assert self.disk >= 0

        self.running_service_instances.append(service)

    def stop_service_instance(self, service):
        self.cpu += service.cpu
        # assert self.cpu <= self.cpu_capacity, "service {} stopped at machine {}".format(service.id, self.get_state())
        self.memory += service.memory
        # assert self.memory <= self.memory_capacity
        self.disk += service.disk
        # assert self.disk <= self.disk_capacity

        self.running_service_instances.remove(service)

    def can_accommodate(self, service_profile):
        if self.destroyed is True:
            return False
        return self.cpu >= service_profile.cpu and \
               self.memory >= service_profile.memory and \
               self.disk >= service_profile.disk

    def compute_failure_score(self, hist_window_size=5):
        # Assume that failure does not happen to the cloud server.
        if self.id == 0:
            return 0

        hist_window_size = min(hist_window_size, len(self.mon_disk_util_hist))
        index = len(self.mon_disk_util_hist) - 1
        cnt = 0
        sum = 0
        while cnt < hist_window_size:
            sum += float(self.mon_disk_util_hist[index])
            index -= 1
            cnt += 1
        return sum / hist_window_size

    def destroy(self):
        services = self.running_service_instances
        for service in services:
            # https://simpy.readthedocs.io/en/latest/simpy_intro/process_interaction.html#interrupting-another-process
            # self.env.interrupt(service)
            service.work_event.interrupt(cause=0)

            # Note that the call for the service interrupt is asynchronous, so update the remaining duration here.
            elapsed_time = service.env.now - service.started_timestamp
            service.duration = service.duration - elapsed_time if elapsed_time <= service.duration else 0

            # self.stop_service_instance(service)
            # self.mec_net.interrupted_services.append(service)

        # FIXME: class 간 dependency 때문에(path cost 연산 등) 해당 머신을 topology 자체에서 지우지는 말고 스케쥴링만 배제되도록 임시 설정해놓음
        # self.mec_net.machines.remove(self)
        self.cpu_capacity = 0
        self.cpu = 0
        self.memory_capacity = 0
        self.memory = 0
        self.disk_capacity = 0
        self.disk = 0
        self.mon_disk_utilization = 0

        self.destroyed = True

    def get_state(self):
        return {
            'id': self.id,
            # 'cpu_capacity': self.cpu_capacity,
            # 'memory_capacity': self.memory_capacity,
            # 'disk_capacity': self.disk_capacity,
            'cpu': "{} / {}".format(self.cpu, self.cpu_capacity),
            'memory': "{} / {}".format(self.memory, self.memory_capacity),
            'disk': "{} / {}".format(self.disk, self.disk_capacity),
            # 'running_task_instances': len(self.running_task_instances),
            # 'finished_task_instances': len(self.finished_task_instances)
        }

    def __repr__(self):
        return str(self.id)


class MachineProfile:

    def __init__(self, cpu_capacity, memory_capacity, disk_capacity, cpu=None, memory=None, disk=None, edgeDC_id=None):
        memory_capacity = round(Decimal(memory_capacity), 9)
        disk_capacity = round(Decimal(disk_capacity), 9)

        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.disk_capacity = disk_capacity

        self.cpu = cpu_capacity if cpu is None else cpu
        self.memory = memory_capacity if memory is None else memory
        self.disk = disk_capacity if disk is None else disk

        self.edgeDC_id = edgeDC_id
