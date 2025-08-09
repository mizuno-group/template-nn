# -*- coding: utf-8 -*-
"""
Created on Aug 9, 2025

@author: tadahaya
"""
from src.utils.general import fix_seed, calc_elapsed_time
import time
import torch

def test_fix_seed_deterministic():
    fix_seed(42)
    tensor1 = torch.rand(3, 3)
    fix_seed(42)
    tensor2 = torch.rand(3, 3)
    assert torch.allclose(tensor1, tensor2)


def test_calc_elapsed_time():
    start_time = time.time()
    time.sleep(1)  # 1秒待つ
    elapsed_time = calc_elapsed_time(start_time)
    assert "0h 0m 1s" in elapsed_time  # 1秒の経過時間が正しく計算されているか確認