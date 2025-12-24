import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def binarize(x):
    return (x > 0).int()

def hamming(a, b):
    return (a != b).sum(axis=1)
