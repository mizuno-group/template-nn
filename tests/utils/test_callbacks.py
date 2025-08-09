# -*- coding: utf-8 -*-
"""

utilのテスト例

@author: tadahaya
"""
from src.utils.metrics import accuracy_score
import torch

def test_accuracy_score():
    y_pred = torch.tensor([1, 0, 2, 1])
    y_true = torch.tensor([1, 1, 2, 0])
    accuracy = accuracy_score(y_pred, y_true)
    assert accuracy == 0.5, "Accuracy score should be correct"