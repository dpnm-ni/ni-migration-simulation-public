import simpy
from core.central_mig_controller import CentralMigrationController
from core.edge_dc import EdgeDC
from core.machine import Machine
from core.mec_net import MECNetwork
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation
from core.monitor import SLAMonitor
from core.injector import FaultInjector
from migration.ACv2.algorithm import ActorCriticv2MigrationAlgorithm
from migration.ACv2.mig_controller import ActorCriticv2MigrationController
from migration.ACv3.algorithm import ActorCriticv3MigrationAlgorithm
from migration.ACv3.mig_controller import ActorCriticv3MigrationController
from migration.ACv4.algorithm import ActorCriticv4MigrationAlgorithm
from migration.ACv4.mig_controller import ActorCriticv4MigrationController
from migration.DQNv2.algorithm import DQNv2MigrationAlgorithm
from migration.DQNv2.mig_controller import DQNv2MigrationController
from migration.DQNv3.algorithm import DQNv3MigrationAlgorithm
from migration.DQNv3.mig_controller import DQNv3MigrationController
from util.config import NUM_EDGE_DC


class Episode:
    def __init__(self, machine_profiles, service_profiles, deployment_algorithm, migration_algorithm):
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

        # Migration controller setup (single-agent vs. multi-agent).
        if isinstance(migration_algorithm, DQNv2MigrationAlgorithm):
            edge_mig_controllers = []
            for i in range(NUM_EDGE_DC + 1):
                edge_mig_controllers.append(DQNv2MigrationController(self.env, migration_algorithm))
            mig_controller = CentralMigrationController(self.env, edge_mig_controllers)

        elif isinstance(migration_algorithm, DQNv3MigrationAlgorithm):
            edge_mig_controllers = []
            for i in range(NUM_EDGE_DC + 1):
                edge_mig_controllers.append(DQNv3MigrationController(self.env, migration_algorithm))
            mig_controller = CentralMigrationController(self.env, edge_mig_controllers)

        # Single-agent ver. for performance comparison.
        elif isinstance(migration_algorithm, ActorCriticv2MigrationAlgorithm):
            mig_controller = ActorCriticv2MigrationController(self.env, migration_algorithm)

        elif isinstance(migration_algorithm, ActorCriticv3MigrationAlgorithm):
            edge_mig_controllers = []
            for i in range(NUM_EDGE_DC + 1):
                edge_mig_controllers.append(ActorCriticv3MigrationController(self.env, migration_algorithm))
            mig_controller = CentralMigrationController(self.env, edge_mig_controllers)

        elif isinstance(migration_algorithm, ActorCriticv4MigrationAlgorithm):
            edge_mig_controllers = []
            for i in range(NUM_EDGE_DC + 1):
                edge_mig_controllers.append(ActorCriticv4MigrationController(self.env, migration_algorithm))
            mig_controller = CentralMigrationController(self.env, edge_mig_controllers)

        else:
            mig_controller = None

        self.simulation = Simulation(self.env, mec_net, service_broker, scheduler, monitor, injector, mig_controller)

    def run(self):
        # Schedule events to be triggered on the SimPy environment.
        self.simulation.run()
        # Run the simulation engine.
        self.env.run()

    def initialize(self):
        EdgeDC.global_index = -1
        Machine.global_index = -1
