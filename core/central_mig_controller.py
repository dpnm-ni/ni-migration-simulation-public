from base_logger import log
from migration.ACv3.algorithm import ActorCriticv3MigrationAlgorithm
from migration.ACv4.algorithm import ActorCriticv4MigrationAlgorithm

# 1/3/5/10/... 각 경우에 성능 평가할 것
MIGRATION_INTERVAL = 10
NUM_ROLLOUT = 10


class CentralMigrationController:
    def __init__(self, env, edge_mig_controllers):
        self.env = env
        assert isinstance(edge_mig_controllers, list)
        self.edge_mig_controllers = edge_mig_controllers
        self.simulation = None
        self.run_cnt = 0

    def attach(self, simulation):
        self.simulation = simulation
        for i in range(len(self.edge_mig_controllers)):
            self.edge_mig_controllers[i].edgeDC_id = i
            self.edge_mig_controllers[i].attach(self.simulation)

    def run(self):
        yield self.env.timeout(MIGRATION_INTERVAL)
        while not self.simulation.is_finished():
            self.run_cnt += 1

            edge_transitions = []
            for edge_controller in self.edge_mig_controllers:
                # Call each agent to have an experience (ideally parallel processing).
                transition = edge_controller.make_migration_decision()
                edge_transitions.append(transition)

            # Make sure to proceed.
            assert len(self.edge_mig_controllers) == len(edge_transitions)

            # Adjust each edge controller's reward in terms of the whole system's reward at this mig tick.
            sum_rewards = 0
            num_involved_edges = 0
            for transition in edge_transitions:
                # None if there are no running services to migrate in the edge at this mig tick.
                if transition is not None:
                    _, _, r, _ = transition
                    sum_rewards += r
                    num_involved_edges += 1

            # avg_reward = sum_rewards / len(edge_transitions) if len(edge_transitions) != 0 else 0
            if num_involved_edges != 0:
                avg_reward = sum_rewards / num_involved_edges
            else:
                # Abnormal case?
                avg_reward = 0

            for i in range(len(edge_transitions)):
                if edge_transitions[i] is not None:
                    # The transition that edge i made.
                    s, a, _, s_prime = edge_transitions[i]

                    # FIXME: an edge controller can access its agent only through the algorithm for now.
                    edge_controller = self.edge_mig_controllers[i]
                    edge_controller.migration_algorithm.agents[i].put_data((s, a, avg_reward, s_prime))
                    edge_controller.hist_rewards.append(avg_reward)

            # FIXME: central controller 각 알고리즘 별로 따로 둘 것
            if isinstance(self.edge_mig_controllers[0].migration_algorithm, ActorCriticv3MigrationAlgorithm):
                if self.run_cnt % NUM_ROLLOUT == 0:
                    for i in range(len(self.edge_mig_controllers)):
                        edge_controller = self.edge_mig_controllers[i]
                        edge_controller.migration_algorithm.agents[i].train()

            if isinstance(self.edge_mig_controllers[0].migration_algorithm, ActorCriticv4MigrationAlgorithm):
                if self.run_cnt % NUM_ROLLOUT == 0:
                    for i in range(len(self.edge_mig_controllers)):
                        edge_controller = self.edge_mig_controllers[i]
                        edge_controller.migration_algorithm.agents[i].train()

            yield self.env.timeout(MIGRATION_INTERVAL)
