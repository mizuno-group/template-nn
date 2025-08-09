# -*- coding: utf-8 -*-
"""
Created on Aug 9, 2025

@author: tadahaya
"""
from src.data.dataset import DataConfig, build_dataloaders

def test_build_dataloaders():
    # テスト用のデータ設定を作成
    data_config = DataConfig(
        batch_size=32,
        num_workers=2,
        train_data_path="data/processed/train",
        val_data_path="data/processed/val",
        test_data_path="data/processed/test"
    )
    
    # DataLoaderをビルド
    train_loader, val_loader, test_loader = build_dataloaders(data_config)
    
    # 各DataLoaderが正しく生成されているか確認
    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None
    
    # バッチサイズが正しいか確認
    assert train_loader.batch_size == 32
    assert val_loader.batch_size == 32
    assert test_loader.batch_size == 32
    
    # ワーカ数が正しいか確認
    assert train_loader.num_workers == 2
    assert val_loader.num_workers == 2
    assert test_loader.num_workers == 2