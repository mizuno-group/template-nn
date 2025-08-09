# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""
import json, os, time, yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


def fix_seed(seed: int=42, fix_cuda: bool=False):
    """
    fix the seed for reproducibility

    Parameters:
    ----------
    seed : int
        the seed number

    """
    # general seed
    random.seed(seed)  # Python random
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)  # PyTorch all GPU seed
    # cudnn seed
    if fix_cuda:
        torch.backends.cudnn.deterministic = True  # for fixing calculation order etc.
        torch.backends.cudnn.benchmark = False  # do not use the optimized algorithm
    # prepare worker seed for DataLoader
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g, seed_worker  # for worker_init_fn in DataLoader


def calc_elapsed_time(start_time):
    """ calculate elapsed time """
    elapsed_time = time.time() - start_time
    h = int(elapsed_time // 3600)
    m = int((elapsed_time % 3600) // 60)
    s = elapsed_time % 60
    return f"{h}h {m}m {s}s"