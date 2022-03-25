from base_logger import log

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_INTERVAL = 1


# TODO: 사실상 agent이므로 brain만 따로 빼고 기능 통합시킬 것
class ActorCriticMigrationController:
    def __init__(self, env, migration_algorithm):
        self.env = env
        self.simulation = None
        self.mec_net = None
        self.migration_algorithm = migration_algorithm

        self.hist_rewards = []

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(MIGRATION_INTERVAL)
        while not self.simulation.is_finished():
            self.make_migration_decision()
            # self.migration_algorithm.agent.train()

            yield self.env.timeout(MIGRATION_INTERVAL)

    def make_migration_decision(self):
        dest_machines, action, reward = self.migration_algorithm(self.mec_net)
        self.migration_algorithm.agent.put_data()
