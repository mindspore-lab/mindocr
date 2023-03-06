'''random seed'''
import random
import numpy as np
import mindspore as ms

def set_seed(seed=42, rank=0):
    '''
    seed: seed int
    rank: rank id
    '''
    if rank is None:
        rank = 0
    random.seed(seed+rank)
    ms.set_seed(seed+rank)
    np.random.seed(seed+rank)