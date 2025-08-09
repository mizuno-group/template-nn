# -*- coding: utf-8 -*-
"""
Created on Aug 9, 2025

@author: tadahaya
"""
import pytest
from src.data.dataset import DataConfig, build_dataloaders
from src.utils.callbacks import EarlyStopping, DefaultLogger

def test_trainer_entry_smoke():
    # trainer 側の公開APIに合わせて変更
    try:
        from src.trainer import Trainer, TrainerConfig
    except ImportError:
        pytest.skip("Trainer API not ready")

    cfg = DataConfig(train_path="data/processed/train", val_path="data/processed/val",
                     batch_size=2, num_workers=0, pin_memory=False)
    train_loader, val_loader = build_dataloaders(cfg)

    class DummyModel:  # 実モデルでも可
        def train(self): pass
        def eval(self): pass
        def to(self, *a, **k): return self

    trainer = Trainer(model=DummyModel(), config=TrainerConfig(), callbacks=[EarlyStopping(), DefaultLogger()], save_dir="outputs")
    # 最低限 fit が呼べるかのみ（内部はモックでOK）
    if hasattr(trainer, "fit"):
        _ = trainer.fit(train_loader, val_loader)
