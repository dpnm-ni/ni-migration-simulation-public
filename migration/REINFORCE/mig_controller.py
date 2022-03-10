from base_logger import log

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_TICK_INTERVAL = 10


class REINFORCEMigrationController:
    def __init__(self, env, migration_algorithm):
        self.env = env
        self.simulation = None
        self.mec_net = None

        self.migration_algorithm = migration_algorithm

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(MIGRATION_TICK_INTERVAL)
        while not self.simulation.is_finished():
            self.make_migration_decision()
            yield self.env.timeout(MIGRATION_TICK_INTERVAL)

    # Find an optimal placement map between all service instances and their host machines every INTERVAL.
    def make_migration_decision(self):
        running_services = self.mec_net.get_unfinished_services()
        for i in range(len(running_services)):
            service = running_services[i]
            state = self.migration_algorithm.get_state(self.mec_net, service)
            if state is None:
                continue

            # Correspond to get_action (and compute_reward) in a general DQN.
            dest_machine, prob_action, reward = self.migration_algorithm(self.mec_net, state)

            src_machine = service.machine
            # !기존 서버에서 기존 서버로의 이전 즉 no migration 케이스는 reward를 다르게 계산할 필요
            # !일단은 로그만 별도로 찍고 동일하게 처리
            if src_machine == dest_machine:
                log.info("[{}] Service {} stays in the current M{}-E{}".format(
                    self.env.now, service.id, src_machine, src_machine.machine_profile.edgeDC_id))

            log.info("[{}] migrate Service {} from M{}-E{} to M{}-E{} (duration: {}, e2e_latency: {}, user_loc: {})".format(
                self.env.now, service.id, src_machine, src_machine.machine_profile.edgeDC_id,
                dest_machine, dest_machine.machine_profile.edgeDC_id,
                service.duration, service.service_profile.e2e_latency, service.user_loc))

            # Apply the live migration decision to the MEC env (corresponding to step in a general DQN).
            service.live_migrate_service_instance(src_machine, dest_machine)
            log.debug("[{}] Source Machine state after: {}".format(self.env.now, src_machine.get_state()))
            log.debug("[{}] Destination Machine state after: {}".format(self.env.now, dest_machine.get_state()))

            self.migration_algorithm.agent.put_data((reward, prob_action))
