import random
import numpy as np
import torch

NUM_EDGE_DC = 15

NUM_EPISODES_ITR = 1


def seed_handler(seed):
    seed = 42
    if seed is not None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
