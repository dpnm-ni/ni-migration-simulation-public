import numpy as np


def default_reward_giver(latency_before, latency_after, availability_before, availability_after):
    # Note: 어떻게 migration 되도 before = 0 보다 좋아질 수 없으므로 reward가 0이다 (X) => -1이다 (현상태 유지 유도)
    if latency_before != 0:
        L_benefit = (latency_before - latency_after) / latency_before
    else:
        if latency_after == 0:
            L_benefit = 0
        else:
            L_benefit = -1

    if availability_before != 0:
        A_benefit = (availability_before - availability_after) / availability_before
    else:
        if availability_after == 0:
            A_benefit = 0
        else:
            A_benefit = -1

    # Apply a non-linear function to each reward.
    # Note: -1, 1 양 극단값에 대해 약 -10, 10으로 펌핑
    L_benefit = np.arctanh(np.clip(L_benefit, -1 + 1e-9, 1 - 1e-9))
    A_benefit = np.arctanh(np.clip(A_benefit, -1 + 1e-9, 1 - 1e-9))

    # FIXME: Apply a weight to each reward (sum to 1?).
    Wl = 1
    Wa = 1

    reward = Wl * L_benefit + Wa * A_benefit

    return reward
