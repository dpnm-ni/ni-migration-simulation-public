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
        # FIXME: interruption delay is ignored
        residence_ratio += (service.finished_timestamp - service.submit_time) / service.duration
    return residence_ratio / len(services)


# compute average queuing delay
def average_queuing_delay(episode):
    queuing_delay = 0
    services = episode.simulation.mec_net.services
    for service in services:
        queuing_delay += (service.started_timestamp - service.queued_timestamp)
    return queuing_delay / len(services)
