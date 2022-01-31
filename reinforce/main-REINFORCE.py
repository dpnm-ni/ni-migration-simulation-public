import time
import numpy as np
from base_logger import log
from core.algorithm import RandomAlgorithm, FirstFitAlgorithm, LeastCostAlgorithm
from core.machine import MachineProfile
from core.episode import Episode
from util.csv_reader import CSVReader
from util.tools import average_completion_time, average_residence_cost, average_queuing_delay
from util.feature_functions import *
from reinforce.agent import REINFORCEAgent
from reinforce.algorithm import REINFORCEAlgorithm
from reinforce.reward_giver import LeastCurrentPathCostRewardGiver, LeastAccumPathCostRewardGiver


# num_machines = 5 if using test_least_cost.csv
NUM_MACHINES = 11

# request file config
# SERVICE_FILE = "csv/test_least_cost.csv"
SERVICE_FILE = "csv/requests_1_3000.csv"
SERVICE_FILE_OFFSET = 0
SERVICE_FILE_LENGTH = 1000

# RL framework config
NUM_ITERATIONS = 100
NUM_EPISODES = 10
# NUM_ROLLOUT = 100
DIM_NN_INPUT = 8


# 시뮬레이터 동작 로그 없이 각 알고리즘의 결과만 출력하려면 base_logger의 로그 레벨 INFO로 바꿀 것
def main():
    machine_profiles = [MachineProfile(64, 1, 1) for _ in range(NUM_MACHINES)]
    service_profiles = CSVReader(SERVICE_FILE, NUM_MACHINES).generate(SERVICE_FILE_OFFSET, SERVICE_FILE_LENGTH)

    """
    # 각 서비스 요청의 엣지 위치와 관계없이 수용가능한 머신들 리스트로 뽑아서 그 중에서 랜덤 선택
    # 비현실적이지만 FirstFit 보다 로드밸런싱 측면에서 나을 수 있음
    # Random
    tic = time.time()
    deployment_algorithm = RandomAlgorithm()
    episode = Episode(machine_profiles, service_profiles, deployment_algorithm)
    episode.run()
    log.info(
        "makespan: {makespan}\n"
        "computation time: {comp_time}\n"
        "avg. completion time: {avg_comp}\n"
        "avg. residence time ratio : {avg_resid}\n"
        "avg. queuing delay: {avg_queue}\n"
        "accum. path cost: {accum_path}\n"
        "avg. path cost: {avg_path}".format(
            makespan=episode.env.now, comp_time=(time.time() - tic), avg_comp=average_completion_time(episode),
            avg_resid=average_residence_cost(episode), avg_queue=average_queuing_delay(episode),
            accum_path=episode.simulation.monitor.accum_path_cost,
            avg_path=np.mean(episode.simulation.monitor.hist_path_costs)))
    log.debug("selected (m, s) pairs: {}".format(episode.simulation.scheduler.valid_pairs))

    # 각 서비스 요청의 엣지 위치와 관계없이 수용가능한 머신들 리스트로 뽑아서 그 중에서 첫번째 pair (서비스, 머신) 선택
    # => 계산 시간은 빠르지만 로드밸런싱이 되지 않아 hot spot 발생으로 인한 성능 저하 가능
    # FirstFit (baseline)
    tic = time.time()
    deployment_algorithm = FirstFitAlgorithm()
    episode = Episode(machine_profiles, service_profiles, deployment_algorithm)
    episode.run()
    log.info(
        "makespan: {makespan}\n"
        "computation time: {comp_time}\n"
        "avg. completion time: {avg_comp}\n"
        "avg. residence time ratio : {avg_resid}\n"
        "avg. queuing delay: {avg_queue}\n"
        "accum. path cost: {accum_path}\n"
        "avg. path cost: {avg_path}".format(
            makespan=episode.env.now, comp_time=(time.time() - tic), avg_comp=average_completion_time(episode),
            avg_resid=average_residence_cost(episode), avg_queue=average_queuing_delay(episode),
            accum_path=episode.simulation.monitor.accum_path_cost,
            avg_path=np.mean(episode.simulation.monitor.hist_path_costs)))
    log.debug("selected (m, s) pairs: {}".format(episode.simulation.scheduler.valid_pairs))

    # 각 서비스 요청의 엣지 머신에 우선 배치 (이 경우 path cost 즉 end latency = 0)
    # 만일 해당 엣지 머신이 해당 서비스 요청을 수용할 수 없는 경우 (리소스 부족 등), path cost가 가장 낮은 다른 인접 머신을 찾아서 거기에 배치
    # => 서비스 latency 측면만 따졌을 때는 해당 알고리즘이 best
    # => 향후 우리 강화학습 알고리즘은 latency 뿐만 아니고 다른 요소(고장 확률, 자원 활용도 등)를 리워드로 고려해서 최적 배치를 학습해야 함
    # LeastCost (better than FF in path cost)
    tic = time.time()
    deployment_algorithm = LeastCostAlgorithm()
    episode = Episode(machine_profiles, service_profiles, deployment_algorithm)
    episode.run()
    log.info(
        "makespan: {makespan}\n"
        "computation time: {comp_time}\n"
        "avg. completion time: {avg_comp}\n"
        "avg. residence time ratio : {avg_resid}\n"
        "avg. queuing delay: {avg_queue}\n"
        "accum. path cost: {accum_path}\n"
        "avg. path cost: {avg_path}".format(
            makespan=episode.env.now, comp_time=(time.time() - tic), avg_comp=average_completion_time(episode),
            avg_resid=average_residence_cost(episode), avg_queue=average_queuing_delay(episode),
            accum_path=episode.simulation.monitor.accum_path_cost,
            avg_path=np.mean(episode.simulation.monitor.hist_path_costs)))
    log.debug("selected (m, s) pairs: {}".format(episode.simulation.scheduler.valid_pairs))
    """


    # # REINFORCE 알고리즘으로 일단 서비스 latency (path cost)만 리워드로 받아 최적 배치 학습
    # # 1. DRL-based deployment hoping to be close to the performance of LeastCost
    agent = REINFORCEAgent(DIM_NN_INPUT)
    # reward_giver = LeastCurrentPathCostRewardGiver()
    reward_giver = LeastAccumPathCostRewardGiver()
    for itr in range(NUM_ITERATIONS):
        log.debug("\n********** Iteration{} ************".format(itr))

        # trajectories = list([])
        makespans = list([])
        computation_times = list([])
        avg_completion_times = list([])
        avg_residence_times = list([])
        avg_queuing_delays = list([])
        accum_path_costs = list([])
        avg_path_costs = list([])
        # for debug purpose
        last_episode = None
        for epi in range(NUM_EPISODES):
            log.debug("\n********** Iteration{} - Episode{} ************".format(itr, epi))
            # initialize everything needed for next episode run
            # s = env.reset()

            tic = time.time()
            algorithm = REINFORCEAlgorithm(agent, reward_giver,
                                    features_extract_func=features_extract_func,
                                    features_normalize_func=features_normalize_func)
            episode = Episode(machine_profiles, service_profiles, algorithm)
            algorithm.reward_giver.attach(episode.simulation)
            episode.run()

            # episode ends
            # trajectories.append(episode.simulation.scheduler.deployment_algorithm.current_trajectory)
            makespans.append(episode.simulation.env.now)
            computation_times.append(time.time() - tic)
            avg_completion_times.append(average_completion_time(episode))
            avg_residence_times.append(average_residence_cost(episode))
            avg_queuing_delays.append(average_queuing_delay(episode))
            accum_path_costs.append(episode.simulation.monitor.accum_path_cost)
            # FIXME:
            avg_path_costs.append(np.mean(episode.simulation.monitor.hist_path_costs))
            last_episode = episode

            observations_epi = []
            actions_epi = []
            rewards_epi = []
            trajectory_epi = episode.simulation.scheduler.deployment_algorithm.current_trajectory
            for transition in trajectory_epi:
                observations_epi.append(transition.observation)
                actions_epi.append(transition.action)
                rewards_epi.append(transition.reward)

            agent.train_net(observations_epi, actions_epi, rewards_epi, episode)
            episode.simulation.scheduler.deployment_algorithm.current_trajectory = []

        # print("\nselected (m, s) pairs: {}".format(last_episode.simulation.scheduler.valid_pairs))
        print(np.mean(makespans), np.mean(computation_times),
              np.mean(avg_completion_times), np.mean(avg_residence_times), np.mean(avg_queuing_delays),
              np.mean(accum_path_costs), np.mean(avg_path_costs))
        # print(np.std(makespans), np.std(computation_times),
        #       np.std(average_completions), np.std(average_slowdowns),
        #       np.std(accum_path_costs), np.std(avg_path_costs))


if __name__ == '__main__':
    main()
