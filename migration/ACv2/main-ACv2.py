import os
import time
from base_logger import log
from core.algorithm import RandomAlgorithm, FirstFitAlgorithm, LeastCostAlgorithm
from core.episode import Episode
from migration.ACv2.agent import ActorCriticv2MigrationAgent
from migration.ACv2.algorithm import ActorCriticv2MigrationAlgorithm
from util.csv_reader import CSVReader
from util.tools import print_result, save_result, write_result
from util.config import NUM_EDGE_DC, seed_handler, NUM_EPISODES_ITR
from util.reward_giver import default_reward_giver

# TODO: 실행 파라미터(topo name) 및 config 파일 이용할 것
# num_machines = 5 if using test_least_cost.csv
# Edgenet: 1 cloud DC and 15 edge DCs.
# NUM_EDGE_DC = 15

# TODO: 테스트 케이스 실행 따로 분리할 것
# SERVICE_FILE = "csv/test_least_cost.csv"
# Make sure reinforce.injector.DISK_FAULT_THRESHOLD = 3 for testing disk_overutil.
# SERVICE_FILE = "csv/test_disk_overutil.csv"
SERVICE_FILE = "csv/requests_1_3000.csv"
SERVICE_FILE_OFFSET = 0
SERVICE_FILE_LENGTH = 1000

# RL config
NUM_ITERATIONS = 3000
# NUM_EPISODES = 1
# DIM_DEP_NN_INPUT = 9
DIM_MIG_NN_INPUT = 8


# 시뮬레이터 동작 로그 없이 각 알고리즘의 결과만 출력하려면 base_logger의 로그 레벨 INFO로 바꿀 것
def main():
    # machine_profiles = [MachineProfile(64, 1, 1) for _ in range(NUM_MACHINES)]
    service_profiles = CSVReader(SERVICE_FILE, NUM_EDGE_DC).generate(SERVICE_FILE_OFFSET, SERVICE_FILE_LENGTH)


    # # Random deployment without migration.
    # tic = time.time()
    # deployment_algorithm = RandomAlgorithm()
    # episode = Episode(None, service_profiles, deployment_algorithm, migration_algorithm=None)
    # episode.run()
    # print_result(episode, tic)
    #
    # # FirstFit deployment without migration.
    # tic = time.time()
    # deployment_algorithm = FirstFitAlgorithm()
    # episode = Episode(None, service_profiles, deployment_algorithm, migration_algorithm=None)
    # episode.run()
    # print_result(episode, tic)
    #
    # # LeastCost deployment without migration.
    # tic = time.time()
    # deployment_algorithm = LeastCostAlgorithm()
    # episode = Episode(None, service_profiles, deployment_algorithm, migration_algorithm=None)
    # episode.run()
    # print_result(episode, tic)


    # Baseline deployment + REINFORCE-based migration.
    # TODO: if DRL-based deployment is also used.
    deployment_agent = None
    migration_agent = ActorCriticv2MigrationAgent(DIM_MIG_NN_INPUT)

    cnt = 0
    for itr in range(NUM_ITERATIONS):
        log.debug("\n********** Iteration{} ************".format(itr))
        for epi in range(NUM_EPISODES_ITR):
            log.debug("\n********** Iteration{} - Episode{} ************".format(itr, epi))
            start_time = time.time()
            deployment_algorithm = FirstFitAlgorithm()
            migration_algorithm = ActorCriticv2MigrationAlgorithm(migration_agent, num_epi=cnt,
                                                                  reward_giver=default_reward_giver)
            episode = Episode(None, service_profiles, deployment_algorithm, migration_algorithm)
            episode.run()

            # Fill the performance measurements at the end of one episode.
            save_result(episode, start_time)

            cnt += 1

        write_result()



if __name__ == '__main__':
    # FIXME:
    os.chdir("../..")

    seed_handler(None)
    main()
