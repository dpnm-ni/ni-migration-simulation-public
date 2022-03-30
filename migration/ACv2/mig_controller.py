from base_logger import log

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_INTERVAL = 30
# NUM_ROLLOUT = 10
NUM_ROLLOUT = 1


# TODO: 사실상 agent이므로 brain만 따로 빼고 기능 통합시킬 것
class ActorCriticv2MigrationController:
    def __init__(self, env, migration_algorithm):
        self.env = env
        self.simulation = None
        self.mec_net = None
        self.migration_algorithm = migration_algorithm

        self.hist_rewards = []
        self.run_cnt = 0

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(MIGRATION_INTERVAL)
        while not self.simulation.is_finished():
            self.run_cnt += 1
            self.make_migration_decision()

            # if len(self.migration_algorithm.agent.data) != 0 and self.run_cnt % NUM_ROLLOUT == 0:
            #     self.migration_algorithm.agent.train()

            yield self.env.timeout(MIGRATION_INTERVAL)

    def make_migration_decision(self):
        transitions = []
        running_services = self.mec_net.get_unfinished_services()
        for i in range(len(running_services)):
            service = running_services[i]
            transition = self.migration_algorithm(self.mec_net, service)
            # FIXME:
            if transition is None:
                continue
            else:
                # s, a, r, s_prime = ret
                # self.migration_algorithm.agent.put_data((s, a, r, s_prime, None))
                # self.hist_rewards.append(r)
                transitions.append(transition)

        sum_rewards = 0
        for transition in transitions:
            _, _, r, _ = transition
            sum_rewards += r
        avg_reward = sum_rewards / len(transitions) if len(transitions) != 0 else 0

        for transition in transitions:
            s, a, _, s_prime = transition
            self.migration_algorithm.agent.put_data((s, a, avg_reward, s_prime, None))
            self.hist_rewards.append(avg_reward)

            self.migration_algorithm.agent.train()
