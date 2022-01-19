import numpy as np
from decimal import Decimal

# machine type: edge or core
class Machine:
    global_index = -1

    def __init__(self, machine_profile):
        self.id = self.get_machine_id()
        self.mec_env = None

        self.machine_profile = machine_profile
        self.cpu_capacity = self.machine_profile.cpu_capacity
        self.memory_capacity = self.machine_profile.memory_capacity
        self.disk_capacity = self.machine_profile.disk_capacity
        self.cpu = self.machine_profile.cpu
        self.memory = self.machine_profile.memory
        self.disk = self.machine_profile.disk

        # FIXME: is this used?
        self.service_instances = []

    @classmethod
    def get_machine_id(cls):
        cls.global_index += 1
        return cls.global_index

    def attach(self, mec_env):
        self.mec_env = mec_env

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

        self.service_instances.append(service)

    def stop_service_instance(self, service):
        self.cpu += service.cpu
        assert self.cpu <= self.cpu_capacity, "service {} stopped at machine {}".format(service.id, self.get_state())
        self.memory += service.memory
        assert self.memory <= self.memory_capacity
        self.disk += service.disk
        assert self.disk <= self.disk_capacity

    def can_accommodate(self, service_profile):
        return self.cpu >= service_profile.cpu and \
               self.memory >= service_profile.memory and \
               self.disk >= service_profile.disk

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


class MachineProfile:
    def __init__(self, cpu_capacity, memory_capacity, disk_capacity, cpu=None, memory=None, disk=None):
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.disk_capacity = disk_capacity

        self.cpu = cpu_capacity if cpu is None else cpu
        self.memory = memory_capacity if memory is None else memory
        self.disk = disk_capacity if disk is None else disk
