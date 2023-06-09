"""random seed"""
import random

import numpy as np

import mindspore as ms


def set_seed(seed=42):
    """
    seed: seed int

    Note: to ensure model init stability, rank_id is removed from seed.
    """
    # if rank is None:
    #    rank = 0
    random.seed(seed)
    ms.set_seed(seed)
    np.random.seed(seed)
