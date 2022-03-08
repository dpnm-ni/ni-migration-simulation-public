from base_logger import log

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_TICK_INTERVAL = 10


class MigrationController:
    def __init__(self, env, migration_algorithm):
        self.env = env
        self.migration_algorithm = migration_algorithm
        self.simulation = None
        self.mec_net = None

    def attach(self, simulation):
        self.simulation = simulation
        self.mec_net = simulation.mec_net

    def run(self):
        yield self.env.timeout(MIGRATION_TICK_INTERVAL)
        while not self.simulation.is_finished():
            self.make_migration_decision()
            yield self.env.timeout(MIGRATION_TICK_INTERVAL)

    # # Find an optimal placement map between all service instances and their host machines every INTERVAL.
    # def make_migration_decision(self):
    #     running_services = self.mec_net.get_unfinished_services()
    #     for i in range(len(running_services)):
    #         service = running_services[i]
    #         # Correspond to get_action in general DQN.
    #         dest_machine, state, action, reward = self.migration_algorithm(self.env.now, self.mec_net, service)
    #
    #         # The service instance stays in the current machine.
    #         if dest_machine is None or reward is None:
    #             # !일단은 기존 서버에서 기존 서버로의 이전 즉 no migration 케이스도 아래 else에서 취급
    #             continue
    #         else:
    #             src_machine = service.machine
    #             if src_machine == dest_machine:
    #                 log.info("[{}] The policy says no migration is needed".format(self.env.now))
    #
    #             log.info("[{}] migrate Service {} from M{}-E{} to M{}-E{} (duration: {})".format(
    #                 self.env.now, service.id, src_machine, src_machine.machine_profile.edgeDC_id,
    #                 dest_machine, dest_machine.machine_profile.edgeDC_id, service.duration))
    #
    #             # Apply the live migration decision to the MEC env (corresponding to step in general DQN).
    #             service.live_migrate_service_instance(src_machine, dest_machine)
    #
    #             log.debug("[{}] Source Machine state after: {}".format(self.env.now, src_machine.get_state()))
    #             log.debug("[{}] Destination Machine state after: {}".format(self.env.now, dest_machine.get_state()))
    #
    #             # FIXME: how to treat next_state and final state?
    #             # !1. Cartpole과 달리 step (migration) 직후 환경으로부터 다음 상태를 바로 확인할 수 없음 (실환경에서 구현이 힘들듯?)
    #             # !일단 next_state <- state로 패딩처리해놨는데 REINFORCE와 달리 DQN에서는 next_state가 학습에 중요한 역할을 한다면 성능 이슈
    #             # !2. final state에 대한 context 있게 만들 것인지? Cartpole 처럼 달성 목표(N회 버티기)가 명확한 문제 아니라면 굳이...
    #             next_state = state
    #             self.migration_algorithm.agent.memorize(state, action, next_state, reward)
    #             self.migration_algorithm.agent.update_q_function()

    # Find an optimal placement map between all service instances and their host machines every INTERVAL.
    def make_migration_decision(self):
        running_services = self.mec_net.get_unfinished_services()
        for i in range(len(running_services)):
            service = running_services[i]
            state = self.migration_algorithm.get_state(self.mec_net, service)
            if state is None:
                continue

            # Correspond to get_action (and compute_reward) in a general DQN.
            dest_machine, action, reward = self.migration_algorithm(self.mec_net, state)

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

            # Get an observation from the env after applying the migration action.
            if i == len(running_services) - 1:
                next_state = state
            else:
                next_state = self.migration_algorithm.get_state(self.mec_net, running_services[i + 1])
            # The migration decision prevents the next service from having a chance to be migrated.
            # !migration 되지 않더라도 기존 머신에서 당분간 동작하는게 크리티컬하진 않으므로 worst reward 까진 아님 (failure 상황이라면 얘기 다름)
            # !일단은 None이 agent로 넘어가면 Tensor 에러나므로 해당 경험은 무효화하고 다음 서비스로
            if next_state is None:
                # Revert the migration decision.
                service.live_migrate_service_instance(dest_machine, src_machine)
                log.info("[{}] cancel migrating Service {}".format(self.env.now, service.id))
                continue

            log.debug("[{}] Source Machine state after: {}".format(self.env.now, src_machine.get_state()))
            log.debug("[{}] Destination Machine state after: {}".format(self.env.now, dest_machine.get_state()))

            self.migration_algorithm.agent.memorize(state, action, next_state, reward)
            self.migration_algorithm.agent.update_q_function()
