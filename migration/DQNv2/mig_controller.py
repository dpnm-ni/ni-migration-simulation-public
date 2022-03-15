from base_logger import log

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_TICK_INTERVAL = 30


class DQNv2MigrationController:
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
                log.info("[{}] No destination machines to migrate Service {}".format(self.env.now, service.id))
                continue

            # Correspond to get_action (and compute_reward) in a general DQN.
            dest_machine, action, reward = self.migration_algorithm(self.mec_net, service)

            src_machine = service.machine
            # !기존 서버에서 기존 서버로의 이전 즉 no migration 케이스는 reward를 다르게 계산할 필요
            # !일단은 로그만 별도로 찍고 동일하게 처리
            if src_machine == dest_machine:
                log.info("[{}] Service {} stays in the current M{}-E{}".format(
                    self.env.now, service.id, src_machine, src_machine.machine_profile.edgeDC_id))

            log.info("[{}] migrate Service {} from M{}-E{} to M{}-E{} (old PC: {}, new PC:{})".format(
                self.env.now, service.id, src_machine, src_machine.machine_profile.edgeDC_id,
                dest_machine, dest_machine.machine_profile.edgeDC_id,
                self.mec_net.get_path_cost(service.user_loc, src_machine.id),
                self.mec_net.get_path_cost(service.user_loc, dest_machine.id)))

            # Apply the live migration decision to the MEC env (corresponding to step in a general DQN).
            service.live_migrate_service_instance(src_machine, dest_machine)

            # FIXME: next state 얻으려고 억지로 구현된 면이 있는데 학습에 꼭 필요한지...
            # Get an observation from the env after applying the migration action.
            if i == len(running_services) - 1:
                # TODO: next_state <- done?
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

            # log.debug("[{}] Source Machine state after: {}".format(self.env.now, src_machine.get_state()))
            # log.debug("[{}] Destination Machine state after: {}".format(self.env.now, dest_machine.get_state()))

            self.migration_algorithm.agent.memorize(state, action, reward, next_state, done_mask=None)
            self.migration_algorithm.agent.train()

            self.hist_rewards.append(reward)
