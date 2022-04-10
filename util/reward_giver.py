import numpy as np


def default_reward_giver(latency_before, latency_after, availability_before, availability_after):
    if latency_before != 0:
        # 감소율이 곧 증가율이므로 곱하기 -1. e.g. (80 - 100) / 100 = -0.2 => 0.2
        L_benefit = (latency_after - latency_before) / latency_before
        L_benefit = -1 * L_benefit
    else:
        # if latency_after == 0:
        #     # before, after 모두 0 => 엣지 favor
        #     L_benefit = ?
        # else:
        #     # before 0, after > 0 => 클라우드 favor
        #     L_benefit = ?
        L_benefit = 0

    if availability_before != 0:
        # e.g. (1 - 0.8) / 0.8 = 0.25
        A_benefit = (availability_after - availability_before) / availability_before
    else:
        # if availability_after == 0:
        #     # before, after 모두 0 => 엣지 favor
        #     A_benefit = ?
        # else:
        #     # before 0, after > 0 => 클라우드 favor
        #     A_benefit = ?
        A_benefit = 0

    # Apply a non-linear function to each reward.
    # Note: -1, 1 양 극단값에 대해 약 -10, 10으로 펌핑
    L_benefit = np.arctanh(np.clip(L_benefit, -1 + 1e-9, 1 - 1e-9))
    A_benefit = np.arctanh(np.clip(A_benefit, -1 + 1e-9, 1 - 1e-9))

    # FIXME: Apply a weight to each reward (sum to 1?).
    Wl = 1
    Wa = 0

    reward = Wl * L_benefit + Wa * A_benefit

    return reward
