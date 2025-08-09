# -*- coding: utf-8 -*-
"""
Created on Aug 9, 2025

@author: tadahaya
"""
import torch
import src.models as models

def test_model_forward_smoke():
    # create_model があればそれを使う、無ければ MyNet を想定
    if hasattr(models, "create_model"):
        model = models.create_model({"name":"mynet", "num_classes":3})
    else:
        model = models.MyNet(output_dim=3)  # 現行のクラス名に合わせてください
    x = torch.randn(2,3,224,224)
    y = model(x)
    assert y.shape[0] == 2