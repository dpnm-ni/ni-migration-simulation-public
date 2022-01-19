def average_completion_time(episode):
    completion_time = 0
    number_services = 0
    for service in episode.simulation.mec_net.services:
        number_services += 1
        completion_time += (service.finished_timestamp - service.started_timestamp)
    return completion_time / number_services


def average_slowdown(episode):
    slowdown = 0
    number_services = 0
    for service in episode.simulation.mec_net.services:
        number_services += 1
        # slowdown += (service.finished_timestamp - service.started_timestamp) / service.duration
        slowdown += (service.finished_timestamp - service.queued_timestamp) / service.duration
    return slowdown / number_services


# def sla_violation(episode):
#     num_violation = 0
#     num_services = 0
#     for service in episode.simulation.mec_net.services:
#         number_services += 1
