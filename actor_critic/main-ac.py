import time

from core.machine import MachineProfile
from core.episode import Episode

from util.csv_reader import CSVReader
from util.tools import average_completion_time, average_slowdown
from util.feature_functions import *

from reinforce.agent import TDActorCritic
from actor_critic.algorithm import RLAlgorithm
from actor_critic.reward_giver import LeastAccumPathCostRewardGiver

# mec net config
NUM_MACHINES = 11

# request file config
# SERVICE_FILE = "csv/test_least_cost.csv"
SERVICE_FILE = "../csv/requests_1_3000.csv"
SERVICE_FILE_OFFSET = 0
SERVICE_FILE_LENGTH = 10

# RL framework config
NUM_ITERATIONS = 100
NUM_EPISODES = 10
# NUM_ROLLOUT = 100
DIM_NN_INPUT = 7


def main():
    machine_profiles = [MachineProfile(64, 1, 1) for _ in range(NUM_MACHINES)]
    service_profiles = CSVReader(SERVICE_FILE, NUM_MACHINES).generate(SERVICE_FILE_OFFSET, SERVICE_FILE_LENGTH)

    # Random
    # tic = time.time()
    # deployment_algorithm = RandomAlgorithm()
    # episode = Episode(machine_profiles, service_profiles, deployment_algorithm)
    # episode.run()
    # print(episode.env.now, time.time() - tic, average_completion_time(episode), average_slowdown(episode),
    #       episode.simulation.monitor.accum_path_cost)

    # FirstFit
    # tic = time.time()
    # deployment_algorithm = FirstFitAlgorithm()
    # episode = Episode(machine_profiles, service_profiles, deployment_algorithm)
    # episode.run()
    # print(episode.env.now, time.time() - tic, average_completion_time(episode), average_slowdown(episode),
    #       episode.simulation.monitor.accum_path_cost)

    # LeastCost is oracle in terms of accumulated E2E latency of all services
    # tic = time.time()
    # deployment_algorithm = LeastCostAlgorithm()
    # episode = Episode(machine_profiles, service_profiles, deployment_algorithm)
    # episode.run()
    # print(episode.env.now, time.time() - tic, average_completion_time(episode), average_slowdown(episode),
    #       episode.simulation.monitor.accum_path_cost)

    # 1. DRL-based deployment hoping to be close to the performance of LeastCost
    agent = TDActorCritic(DIM_NN_INPUT)
    reward_giver = LeastAccumPathCostRewardGiver()
    for itr in range(NUM_ITERATIONS):
        tic = time.time()

        trajectories = list([])
        makespans = list([])
        average_completions = list([])
        average_slowdowns = list([])
        accum_path_costs = list([])
        for epi in range(NUM_EPISODES):
            print("\n********** itr{}-epi{} ************".format(itr, epi))
            # initialize everything needed for next episode run
            # s = env.reset()

            algorithm = RLAlgorithm(agent, reward_giver,
                                    features_extract_func=features_extract_func,
                                    features_normalize_func=features_normalize_func)
            episode = Episode(machine_profiles, service_profiles, algorithm)
            algorithm.reward_giver.attach(episode.simulation)
            episode.run()

            trajectories.append(episode.simulation.scheduler.deployment_algorithm.current_trajectory)
            makespans.append(episode.simulation.env.now)
            average_completions.append(average_completion_time(episode))
            average_slowdowns.append(average_slowdown(episode))
            accum_path_costs.append(episode.simulation.monitor.accum_path_cost)

            observations_epi = []
            actions_epi = []
            rewards_epi = []
            trajectory_epi = episode.simulation.scheduler.deployment_algorithm.current_trajectory
            # FIXME: change node to transition
            for node in trajectory_epi:
                observations_epi.append(node.observation)
                actions_epi.append(node.action)
                rewards_epi.append(node.reward)

            agent.train_net(observations_epi, actions_epi, rewards_epi)
            episode.simulation.scheduler.deployment_algorithm.current_trajectory = []


        print(np.mean(makespans), time.time() - tic,
              np.mean(average_completions), np.mean(average_slowdowns), np.mean(accum_path_costs))

        # all_observations = []
        # all_actions = []
        # all_rewards = []
        # for trajectory_epi in trajectories:
        #     observations = []
        #     actions = []
        #     rewards = []
        #     for node in trajectory_epi:
        #         observations.append(node.observation)
        #         actions.append(node.action)
        #         rewards.append(node.reward)
        #
        #     all_observations.append(observations)
        #     all_actions.append(actions)
        #     all_rewards.append(rewards)
        #
        # # all_q_s, all_advantages = agent.estimate_return(all_rewards)
        # # agent.update_parameters(all_observations, all_actions, all_advantages)
        # agent.train_net(all_observations, all_actions, all_rewards)

        # done = False
        # while not done:
        #     for _ in range(NUM_ROLLOUT):
        #         prob = agent.pi(torch.from_numpy(s).float())
        #         m = Categorical(prob)
        #         # 정책넷 PI의 확률분포 P(a|s)에 따라 액션 샘플링
        #         a = m.sample().item()
        #         s_prime, r, done, info = env.step(a)
        #         model.put_data((s, a, r, s_prime, done))  # 데이터 저장 단위: transition (s,a,r,s')
        #
        #         s = s_prime
        #         score += r
        #
        #         if done:
        #             break
        #
        #             # n_rollout * step 단위로 뉴럴넷 업데이트 => non-episodic MDP에 적용 가능
        #     model.train_net()
        #
        # if n_epi % print_interval == 0 and n_epi != 0:
        #     print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
        #     score = 0.0

    # 2. FirstFit deployment and DRL-based migration hoping to be best at failure events


if __name__ == '__main__':
    main()
