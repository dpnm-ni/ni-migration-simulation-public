import time

from core.machine import MachineProfile
from core.algorithm import *
from core.episode import Episode
from util.csv_reader import CSVReader
from util.tools import average_completion_time, average_slowdown

MACHINE_NUMBER = 13
# SERVICE_FILE = "csv/test_least_cost.csv"
SERVICE_FILE = "csv/requests_3000.csv"
SERVICE_FILE_OFFSET = 0
SERVICE_FILE_LENGTH = 3000


def main():
    machine_profiles = [MachineProfile(64, 1, 1) for i in range(MACHINE_NUMBER)]
    service_profiles = CSVReader(SERVICE_FILE).generate(SERVICE_FILE_OFFSET, SERVICE_FILE_LENGTH)

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
    tic = time.time()
    deployment_algorithm = LeastCostAlgorithm()
    episode = Episode(machine_profiles, service_profiles, deployment_algorithm)
    episode.run()
    print(episode.env.now, time.time() - tic, average_completion_time(episode), average_slowdown(episode),
          episode.simulation.monitor.accum_path_cost)

    # 1. DRL-based deployment hoping to be close to LeastCost

    # 2. FirstFit deployment and DRL-based migration hoping to be best at failure events


if __name__ == '__main__':
    main()
