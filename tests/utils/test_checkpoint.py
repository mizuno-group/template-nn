# -*- coding: utf-8 -*-
"""
Created on Aug 9, 2025

@author: tadahaya
"""
import torch
from torch import nn
from src.utils.checkpoint import save_experiment, load_experiment

def test_save_load_experiment(tmp_path):
    # モデルとオプティマイザの定義
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # ダミーデータ
    dummy_data = torch.randn(5, 10)
    dummy_target = torch.randn(5, 1)
    
    # 学習ステップの実行
    model.train()
    optimizer.zero_grad()
    output = model(dummy_data)
    loss = nn.MSELoss()(output, dummy_target)
    loss.backward()
    optimizer.step()
    
    # チェックポイントの保存
    save_experiment(model, optimizer, tmp_path / "checkpoint.pth")
    
    # チェックポイントの読み込み
    loaded_model, loaded_optimizer = load_experiment(tmp_path / "checkpoint.pth")
    
    # モデルとオプティマイザが正しく読み込まれたか確認
    assert isinstance(loaded_model, nn.Linear)
    assert isinstance(loaded_optimizer, torch.optim.Adam)