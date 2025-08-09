# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

utils

@author: tadahaya
"""

from typing import Dict, Optional

import json, os, time, yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


def progress_plot(
        outdir:str, train_values:list, test_values:list=[],
        xlabel="epoch", ylabel="loss"
        ):
    """ plot learning progress """
    fileout = os.path.join(outdir, f"progress_{ylabel}.tif")
    x = list(range(1, len(train_values) + 1, 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 14
    ax.plot(x, train_values, c='navy', label='train')
    if len(test_values) > 0:
        ax.plot(x, test_values, c='darkgoldenrod', label='test')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(fileout, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
