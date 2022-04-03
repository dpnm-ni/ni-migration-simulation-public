from base_logger import log

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_INTERVAL = 20
NUM_ROLLOUT = 10


# TODO: 사실상 agent이므로 brain만 따로 빼고 기능 통합시킬 것
class ActorCriticv3MigrationController:
    def __init__(self, env, migration_algorithm):
        self.env = env
        self.simulation = None
        self.migration_algorithm = migration_algorithm

        self.mec_net = None
        self.edgeDC_id = None

        self.hist_rewards = []
        self.run_cnt = 0

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(MIGRATION_INTERVAL)
        while not self.simulation.is_finished():
            self.run_cnt += 1
            # print(self.edgeDC_id)
            self.make_migration_decision()

            if len(self.migration_algorithm.agents[self.edgeDC_id].data) != 0 and self.run_cnt % NUM_ROLLOUT == 0:
                self.migration_algorithm.agents[self.edgeDC_id].train()

            yield self.env.timeout(MIGRATION_INTERVAL)

    def make_migration_decision(self):
        transition = self.migration_algorithm(self.mec_net, self.edgeDC_id)
        if transition is None:
            return
        else:
            s, a, r, s_prime = transition
            self.migration_algorithm.agents[self.edgeDC_id].put_data((s, a, r, s_prime, None))

            self.hist_rewards.append(r)
