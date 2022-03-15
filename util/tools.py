# FIXME: remove all unused functions.
import time
import numpy as np
from base_logger import log

NUM_EPISODES_ITR = 10
sim_times_itr = [] * NUM_EPISODES_ITR
makespans_itr = [] * NUM_EPISODES_ITR
avg_resource_utils_itr = [] * NUM_EPISODES_ITR
service_interruptions_itr = [] * NUM_EPISODES_ITR
accum_path_costs_itr = [] * NUM_EPISODES_ITR
avg_service_latencies_itr = [] * NUM_EPISODES_ITR
avg_service_prov_delays_itr = [] * NUM_EPISODES_ITR
accum_rewards_itr = [] * NUM_EPISODES_ITR
avg_rewards_itr = [] * NUM_EPISODES_ITR


# compute average time for a service from its submission to completion
def average_completion_time(episode):
    completion_time = 0
    services = episode.simulation.mec_net.services
    for service in services:
        completion_time += (service.finished_timestamp - service.submit_time)
    return completion_time / len(services)


# compute average ratio for a service's residence time
# effective as the ratio gets closer to 1
def average_residence_cost(episode):
    residence_ratio = 0
    services = episode.simulation.mec_net.services
    for service in services:
        residence_ratio += (service.finished_timestamp - service.submit_time) / service.duration
    return residence_ratio / len(services)


# compute average queuing delay
def average_queuing_delay(episode):
    queuing_delay = 0
    services = episode.simulation.mec_net.services
    for service in services:
        queuing_delay += (service.started_timestamp - service.queued_timestamp)
    return queuing_delay / len(services)


def print_result(episode, start_time):
    log.info(
        "sim execution time: {sim_time}\n"
        "makespan: {makespan}\n"
        "avg. DC resource util: {avg_resource_util}\n"
        "service interruption: {service_interruption}\n"
        "accumulated path cost: {accum_path_cost}\n"
        "avg. service latency: {avg_service_latency}\n"
        "avg. service provisioning delay: {avg_service_prov_delay}".format(
            sim_time=(time.time() - start_time),
            makespan=episode.env.now,
            avg_resource_util=average_resource_utilization(episode),
            service_interruption=service_interruptions(episode),
            accum_path_cost=episode.simulation.monitor.accum_path_cost,
            avg_service_latency=average_service_latency(episode),
            avg_service_prov_delay=average_service_queuing_delay(episode)
        )
    )


def save_result(episode, start_time):
    sim_times_itr.append(time.time() - start_time)
    makespans_itr.append(episode.env.now)
    # Take cpu util only.
    avg_resource_utils_itr.append([edge_util[0] for edge_util in average_resource_utilization(episode)])
    service_interruptions_itr.append(service_interruptions(episode))
    accum_path_costs_itr.append(episode.simulation.monitor.accum_path_cost)
    avg_service_latencies_itr.append(average_service_latency(episode))
    avg_service_prov_delays_itr.append(average_service_queuing_delay(episode))

    accum_rewards_itr.append(np.sum(episode.simulation.controller.hist_rewards))
    avg_rewards_itr.append(np.mean(episode.simulation.controller.hist_rewards))


def write_result():
    with open("result.txt", 'a') as f:
        print(str(np.mean(sim_times_itr)) + ", " +
              str(np.mean(makespans_itr)) + ", " +
              str(np.mean(avg_resource_utils_itr, axis=0)) + ", " +
              str(np.mean(service_interruptions_itr)) + ", " +
              str(np.mean(accum_path_costs_itr)) + ", " +
              str(np.mean(avg_service_latencies_itr, axis=0)) + ", " +
              str(np.mean(avg_service_prov_delays_itr, axis=0)) + ", " +
              str(np.mean(accum_rewards_itr)) + ", " +
              str(np.mean(avg_rewards_itr)), file=f)

    sim_times_itr.clear()
    makespans_itr.clear()
    avg_resource_utils_itr.clear()
    service_interruptions_itr.clear()
    accum_path_costs_itr.clear()
    avg_service_latencies_itr.clear()
    avg_service_prov_delays_itr.clear()
    accum_rewards_itr.clear()
    avg_rewards_itr.clear()


def average_resource_utilization(episode):
    num_edgeDCs = len(episode.simulation.mec_net.edgeDCs)
    edgeDCs_cpu_util = [[] for _ in range(num_edgeDCs)]
    edgeDCs_memory_util = [[] for _ in range(num_edgeDCs)]
    edgeDCs_disk_util = [[] for _ in range(num_edgeDCs)]

    machines = episode.simulation.mec_net.machines
    for machine in machines:
        if machine.destroyed is True:
            continue
        avg_cpu_util = np.mean(machine.mon_cpu_util_hist)
        avg_memory_util = float(np.mean(machine.mon_memory_util_hist))
        avg_disk_util = float(np.mean(machine.mon_disk_util_hist))

        edgeDC_id = machine.machine_profile.edgeDC_id
        edgeDCs_cpu_util[edgeDC_id].append(avg_cpu_util)
        edgeDCs_memory_util[edgeDC_id].append(avg_memory_util)
        edgeDCs_disk_util[edgeDC_id].append(avg_disk_util)

    # Take average over all machines in each edgeDC.
    for edgeDC_id in range(num_edgeDCs):
        edgeDCs_cpu_util[edgeDC_id] = np.mean(edgeDCs_cpu_util[edgeDC_id])
        edgeDCs_memory_util[edgeDC_id] = np.mean(edgeDCs_memory_util[edgeDC_id])
        edgeDCs_disk_util[edgeDC_id] = np.mean(edgeDCs_disk_util[edgeDC_id])

    # result = []
    # for edgeDC_util in zip(edgeDCs_cpu_util, edgeDCs_memory_util, edgeDCs_disk_util):
    #     result.append([np.mean(edgeDC_util[0]), np.mean(edgeDC_util[1]), np.mean(edgeDC_util[2])])
    # return result
    result = []
    # result[0]: cloud DC util
    result.append([edgeDCs_cpu_util[0], edgeDCs_memory_util[0], edgeDCs_disk_util[0]])
    # result[1]: edge DC util averaged over all edge DCs
    result.append([np.mean(edgeDCs_cpu_util[1:]), np.mean(edgeDCs_memory_util[1:]), np.mean(edgeDCs_disk_util[1:])])
    return result


def service_interruptions(episode):
    total = 0
    for service in episode.simulation.mec_net.services:
        total += service.num_interruptions_by_fault
    return total


def average_service_latency(episode):
    result = []
    for hist_type_path_cost in episode.simulation.monitor.hist_types_path_costs:
        result.append(np.mean(hist_type_path_cost))
    return result


# FIXME:
def average_service_queuing_delay(episode):
    sum_types_qdelay = [0] * 4
    cnt_types = [0] * 4
    services = episode.simulation.mec_net.services
    for service in services:
        sum_types_qdelay[service.get_service_type()] += (service.started_timestamp - service.queued_timestamp)
        cnt_types[service.get_service_type()] += 1

    result = []
    for sum_type_qdelay, cnt_type in zip(sum_types_qdelay, cnt_types):
        avg_type_qdelay = sum_type_qdelay / cnt_type
        result.append(avg_type_qdelay)
    return result
