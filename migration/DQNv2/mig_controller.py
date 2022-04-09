from base_logger import log

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_INTERVAL = 10


class DQNv2MigrationController:
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

    def run(self):
        yield self.env.timeout(MIGRATION_INTERVAL)
        while not self.simulation.is_finished():
            # print(self.edgeDC_id)
            self.make_migration_decision()

            # # TODO: train 실행 시점, 횟수 주의, 위치 변경
            # self.migration_algorithm.agents[self.edgeDC_id].train()

            yield self.env.timeout(MIGRATION_INTERVAL)

    def make_migration_decision(self):
        transition = self.migration_algorithm(self.mec_net, self.edgeDC_id)
        if transition is None:
            return
        else:
            s, a, r, s_prime = transition
            # TODO: agent 개별 메모리 또는 공유 메모리? 일단은 전자
            self.migration_algorithm.agents[self.edgeDC_id].memorize(s, a, r, s_prime)

            self.hist_rewards.append(r)
