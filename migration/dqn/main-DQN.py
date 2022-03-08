import os
import time
import numpy as np
from config import cfg
from base_logger import log
from core.algorithm import RandomAlgorithm, FirstFitAlgorithm, LeastCostAlgorithm
from core.episode import Episode
from migration.dqn.agent import DQNMigrationAgent
from migration.dqn.algorithm import DQNMigrationAlgorithm
from util.csv_reader import CSVReader
from util.tools import average_completion_time, average_residence_cost, average_queuing_delay


# TODO: 실행 파라미터(topo name) 및 config 파일 이용할 것
# num_machines = 5 if using test_least_cost.csv
# Edgenet: 1 cloud DC and 15 edge DCs.
NUM_EDGE_DC = 15

# TODO: 테스트 케이스 실행 따로 분리할 것
# SERVICE_FILE = "csv/test_least_cost.csv"
# Make sure reinforce.injector.DISK_FAULT_THRESHOLD = 3 for testing disk_overutil.
# SERVICE_FILE = "csv/test_disk_overutil.csv"
SERVICE_FILE = "csv/requests_1_3000.csv"
SERVICE_FILE_OFFSET = 0
SERVICE_FILE_LENGTH = 1000

# RL config
NUM_ITERATIONS = 100
NUM_EPISODES = 10
DIM_DEP_NN_INPUT = 9
DIM_MIG_NN_INPUT = 7


# 시뮬레이터 동작 로그 없이 각 알고리즘의 결과만 출력하려면 base_logger의 로그 레벨 INFO로 바꿀 것
def main():
    # machine_profiles = [MachineProfile(64, 1, 1) for _ in range(NUM_MACHINES)]
    service_profiles = CSVReader(SERVICE_FILE, NUM_EDGE_DC).generate(SERVICE_FILE_OFFSET, SERVICE_FILE_LENGTH)

    # Random deployment without migration.
    tic = time.time()
    deployment_algorithm = RandomAlgorithm()
    episode = Episode(None, service_profiles, deployment_algorithm)
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

    # FirstFit deployment without migration.
    tic = time.time()
    deployment_algorithm = FirstFitAlgorithm()
    episode = Episode(None, service_profiles, deployment_algorithm, migration_algorithm=None)
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

    # LeastCost deployment without migration.
    tic = time.time()
    deployment_algorithm = LeastCostAlgorithm()
    episode = Episode(None, service_profiles, deployment_algorithm, migration_algorithm=None)
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


    # FirstFit deployment with DQN-based migration.
    # !체크리스트
    # !1차: LeastCost deployment에 필적하는 성능
    # !2차: 고장 상황 시에 LeastCost deployment 보다 향상된 성능
    cnt = 0
    for itr in range(NUM_ITERATIONS):
        log.debug("\n********** Iteration{} ************".format(itr))

        # Performance metrics that are updated every iteration.
        makespans = list([])
        computation_times = list([])
        avg_completion_times = list([])
        avg_residence_times = list([])
        avg_queuing_delays = list([])
        accum_path_costs = list([])
        avg_path_costs = list([])
        # !disk_overutil 리워드 1일 때가 0.5, 0보다 interruption 개수 크게 나옴...이상한데 failure 랜덤성 때문인가?
        avg_num_interruptions = 0

        for epi in range(NUM_EPISODES):
            log.debug("\n********** Iteration{} - Episode{} ************".format(itr, epi))
            tic = time.time()
            # TODO: if DRL-based deployment is also used.
            deployment_agent = None
            deployment_algorithm = FirstFitAlgorithm()
            migration_agent = DQNMigrationAgent(DIM_MIG_NN_INPUT, decaying_cnt=cnt)
            migration_algorithm = DQNMigrationAlgorithm(migration_agent)
            episode = Episode(None, service_profiles, deployment_algorithm, migration_algorithm)
            episode.run()

            # Fill the performance measurements at the end of one episode.
            makespans.append(episode.simulation.env.now)
            computation_times.append(time.time() - tic)
            avg_completion_times.append(average_completion_time(episode))
            avg_residence_times.append(average_residence_cost(episode))
            avg_queuing_delays.append(average_queuing_delay(episode))
            # FIXME: compute at utils like other metric.
            accum_path_costs.append(episode.simulation.monitor.accum_path_cost)
            avg_path_costs.append(np.mean(episode.simulation.monitor.hist_path_costs))
            avg_num_interruptions += len(episode.simulation.mec_net.interrupted_services)

            # observations_epi = []
            # actions_epi = []
            # rewards_epi = []
            # trajectory_epi = episode.simulation.scheduler.deployment_algorithm.current_trajectory
            # for transition in trajectory_epi:
            #     observations_epi.append(transition.observation)
            #     actions_epi.append(transition.action)
            #     rewards_epi.append(transition.reward)
            # migration_agent.train_net(observations_epi, actions_epi, rewards_epi, episode)
            # episode.simulation.scheduler.deployment_algorithm.current_trajectory = []

            # Sync the target QNet (DNN) with the main (learning) QNet every episode.
            migration_agent.update_target_q_function()

            # !찍었을 때 누적 보상이 증가하지는 않음을 확인함
            # cum_reward_epi = 0
            # for reward in rewards_epi:
            #     if reward is not None:
            #         cum_reward_epi += reward
            # print(cum_reward_epi)

            cnt += 1

        with open("result.txt", 'a') as f:
            print(np.mean(makespans), np.mean(computation_times),
                  np.mean(avg_completion_times), np.mean(avg_residence_times), np.mean(avg_queuing_delays),
                  np.mean(accum_path_costs), np.mean(avg_path_costs), np.mean(avg_num_interruptions), file=f)



if __name__ == '__main__':
    # FIXME:
    os.chdir("../..")
    main()
