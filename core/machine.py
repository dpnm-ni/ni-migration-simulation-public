import numpy as np
from decimal import Decimal


# machine type: edge or core
class Machine:
    global_index = -1

    def __init__(self, machine_profile):
        # self.id = self.get_machine_id()
        self.id = machine_profile.machine_id
        self.mec_net = None

        self.machine_profile = machine_profile
        self.cpu_capacity = self.machine_profile.cpu_capacity
        self.memory_capacity = self.machine_profile.memory_capacity
        self.disk_capacity = self.machine_profile.disk_capacity
        self.cpu = self.machine_profile.cpu
        self.memory = self.machine_profile.memory
        self.disk = self.machine_profile.disk

        # periodically filled by Monitor
        self.mon_disk_utilization = 0
        self.mon_disk_overutil_cnt = 0

        self.running_service_instances = []
        self.destroyed = False

    @classmethod
    def get_machine_id(cls):
        cls.global_index += 1
        return cls.global_index

    def attach(self, mec_net):
        self.mec_net = mec_net

    # Migrating instanceA from server1 to server2 is simulated by:
    # if server2.can_accommodate(instanceA)
    #   server1.stop_container_instance(instanceA)
    #   server2.run_container_instance(instanceA)
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

    def destroy(self):
        services = self.running_service_instances
        for service in services:
            # https://simpy.readthedocs.io/en/latest/simpy_intro/process_interaction.html#interrupting-another-process
            # self.env.interrupt(service)
            service.work_event.interrupt()
            # FIXME: for문 iterable(services)에 대한 remove 연산 때문에 일부 서비스가 interrupt 되지 않음. 유사한 코드 전체적으로 확인 필요
            # self.stop_service_instance(service)
            self.mec_net.interrupted_services.append(service)

        # !중요: class 간 dependency 때문에(path cost 연산 등) 해당 머신을 topology 자체에서 지우지는 말고 스케쥴링만 배제되도록 임시 설정해놓음
        # self.mec_net.machines.remove(self)
        # self.cpu_capacity = 0
        # self.memory_capacity = 0
        # self.disk_capacity = 0
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
    global_index = 0

    def __init__(self, cpu_capacity, memory_capacity, disk_capacity, cpu=None, memory=None, disk=None):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.disk_capacity = disk_capacity

        self.cpu = cpu_capacity if cpu is None else cpu
        self.memory = memory_capacity if memory is None else memory
        self.disk = disk_capacity if disk is None else disk

        self.machine_id = MachineProfile.global_index
        MachineProfile.global_index += 1
