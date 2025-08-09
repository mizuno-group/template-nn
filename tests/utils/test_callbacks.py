# -*- coding: utf-8 -*-
"""
Created on Aug 9, 2025

@author: tadahaya
"""
from src.utils.callbacks import EarlyStopping, DefaultLogger

def test_early_stopping():
    # EarlyStoppingのインスタンスを作成
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    # ダミーの損失値を設定
    losses = [0.5, 0.4, 0.35, 0.33, 0.32, 0.31, 0.3]
    
    # 各エポックでのEarlyStoppingの状態を確認
    for epoch, loss in enumerate(losses):
        should_stop = early_stopping.step(loss)
        if epoch < 3:
            assert not should_stop, f"Epoch {epoch} should not trigger early stopping."
        else:
            assert should_stop, f"Epoch {epoch} should trigger early stopping."


def test_default_logger(tmp_path):
    # DefaultLoggerのインスタンスを作成
    logger = DefaultLogger(save_dir=str(tmp_path), log_interval=1)
    
    # ダミーのログデータを設定
    epoch = 1
    metrics = {'loss': 0.5, 'accuracy': 0.8}
    
    # ログを記録
    logger.log(epoch, metrics)
    
    # ログファイルが作成されたか確認
    log_file = tmp_path / "default_log.txt"
    assert log_file.is_file()
    
    # ログ内容を確認
    with open(log_file, 'r') as f:
        content = f.read()
        assert "Epoch: 1" in content
        assert "loss: 0.5" in content
        assert "accuracy: 0.8" in content