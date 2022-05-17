# FIXME: 별다른 기능 없어짐. agent와 통합할 것
class DQNv3MigrationController:
    def __init__(self, env, migration_algorithm):
        self.env = env
        self.simulation = None
        self.migration_algorithm = migration_algorithm

        self.mec_net = None
        self.edgeDC_id = None

        self.hist_rewards = []

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def make_migration_decision(self):
        transition = self.migration_algorithm(self.mec_net, self.edgeDC_id)
        return transition
