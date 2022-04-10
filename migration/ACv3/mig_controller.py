from base_logger import log

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_INTERVAL = 10
NUM_ROLLOUT = 10


# FIXME: 별다른 기능 없어짐. agent와 통합할 것
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

    # def run(self):
    #     return self.make_migration_decision()

    def make_migration_decision(self):
        transition = self.migration_algorithm(self.mec_net, self.edgeDC_id)

        return transition


        # if transition is None:
        #     return
        # else:
        #     s, a, r, s_prime = transition
        #     self.migration_algorithm.agents[self.edgeDC_id].put_data((s, a, r, s_prime))
        #
        #     self.hist_rewards.append(r)
