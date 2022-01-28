import numpy as np


def features_extract_func(service):
    return [service.service_profile.cpu, service.service_profile.memory,
            service.service_profile.disk, service.service_profile.duration]


# def features_extract_func_ac(task):
#     return features_extract_func(task) + [task.task_config.instances_number, len(task.running_task_instances),
#                                           len(task.finished_task_instances)]

# FIXME: constants only calculated about requests_1_3000.csv
def features_normalize_func(x):
    # x: list of [m.cpu, m.mem, m.disk, s.cpu, s.mem, s.disk, s.dur, m-s path cost]
    # min-max ver. (X-MIN)/(MAX-MIN)
    y = (np.array(x).astype(float) - np.array([0, 0, 0, 1, 0.002651, 0.000142, 1])) / np.array([64, 1, 1, 15, 0.997349, 0.113475, 3627.828])
    # z-score ver. (X-MEAN)/STD
    # y = (x - np.array(np.mean(x))) / np.array(np.std(x))

    return y


# def features_normalize_func_ac(x):
#     y = (np.array(x) - np.array([0, 0, 0.65, 0.009, 74.0, 80.3, 80.3, 80.3, 80.3])) / np.array(
#         [64, 1, 0.23, 0.005, 108.0, 643.5, 643.5, 643.5, 643.5])
#     return y
