import simpy
from core.edge_dc import EdgeDC
from core.machine import Machine
from core.mec_net import MECNetwork
from core.scheduler import Scheduler
from core.broker import Broker
from core.simulation import Simulation
from core.monitor import SLAMonitor
from core.injector import FaultInjector
from migration.AC.algorithm import ActorCriticMigrationAlgorithm
from migration.AC.mig_controller import ActorCriticMigrationController
from migration.ACv2.algorithm import ActorCriticv2MigrationAlgorithm
from migration.ACv2.mig_controller import ActorCriticv2MigrationController
from migration.ACv3.algorithm import ActorCriticv3MigrationAlgorithm
from migration.ACv3.mig_controller import ActorCriticv3MigrationController
from migration.DQNv2.algorithm import DQNv2MigrationAlgorithm
from migration.DQNv2.mig_controller import DQNv2MigrationController
from migration.REINFORCE.algorithm import REINFORCEMigrationAlgorithm
from migration.REINFORCE.mig_controller import REINFORCEMigrationController
from migration.DQN.algorithm import DQNMigrationAlgorithm
from migration.DQN.mig_controller import DQNMigrationController


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

        if isinstance(migration_algorithm, DQNMigrationAlgorithm):
            mig_controller = DQNMigrationController(self.env, migration_algorithm)
        elif isinstance(migration_algorithm, DQNv2MigrationAlgorithm):
            mig_controller = []
            # FIXME:
            for i in range(16):
                mig_controller.append(DQNv2MigrationController(self.env, migration_algorithm))
        elif isinstance(migration_algorithm, REINFORCEMigrationAlgorithm):
            mig_controller = REINFORCEMigrationController(self.env, migration_algorithm)
        elif isinstance(migration_algorithm, ActorCriticMigrationAlgorithm):
            mig_controller = ActorCriticMigrationController(self.env, migration_algorithm)
        elif isinstance(migration_algorithm, ActorCriticv2MigrationAlgorithm):
            mig_controller = ActorCriticv2MigrationController(self.env, migration_algorithm)
        elif isinstance(migration_algorithm, ActorCriticv3MigrationAlgorithm):
            mig_controller = []
            # FIXME:
            for i in range(16):
                mig_controller.append(ActorCriticv3MigrationController(self.env, migration_algorithm))
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
