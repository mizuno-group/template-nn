# -*- coding: utf-8 -*-
"""
Created on Friday August 15 15:41:44 2025

プロジェクト全体で使う型エイリアスをここに集約

@author: tadahaya
"""
from typing import Union
import numpy as np
import torch

ArrayLike = Union[np.ndarray, "torch.Tensor"]