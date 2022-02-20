import simpy

from core.edge_dc import EdgeDC
from core.machine import Machine
from core.mec_net import MECNetwork
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation
from core.monitor import SLAMonitor
from migration.reinforce.injector import FaultInjector


class Episode:
    def __init__(self, machine_profiles, service_profiles, deployment_algorithm):
        self.initialize()

        self.env = simpy.Environment()
        # self.env = simpy.RealtimeEnvironment()

        mec_net = MECNetwork()
        # mec_net.add_machines(machine_profiles)
        mec_net.create_edgeDCs()
        mec_net.apply_edgeDCs()

        service_broker = Broker(self.env, service_profiles)
        scheduler = Scheduler(self.env, deployment_algorithm)
        monitor = SLAMonitor(self.env)
        injector = FaultInjector(self.env)

        self.simulation = Simulation(self.env, mec_net, service_broker, scheduler, monitor, injector)

    def run(self):
        # run our simulation environment
        # analogy to env.process(simulation(env)) for event scheduling
        self.simulation.run()
        # run simpy simulation environment to trigger scheduled events
        self.env.run()

    def initialize(self):
        EdgeDC.global_index = -1
        Machine.global_index = -1
