# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

学習のentry point
- YAMDLファイルを読み込んで学習を開始する

@author: tadahaya
"""
from __future__ import annotations
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch

from src.data.dataset import DataConfig, build_dataloaders
import src.models as models
from src.trainer import Trainer, TrainerConfig
from src.utils.general import fix_seed, calc_elapsed_time
import src.utils.callbacks as callback_module # 可読性向上のためエイリアスを使用


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する関数.
    
    Returns
    -------
    argparse.Namespace
        解析された引数を含むNamespaceオブジェクト.
    
    """
    parser = argparse.ArgumentParser(description="Train entry")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to the YAML configuration file.")
    return parser.parse_args()


def run_training(
    model,
    train_loader,
    val_loader,
    trainer_cfg: TrainerConfig,
    callbacks: list[Any],
    save_dir: str,
    ) -> Dict[str, Any]:
    """
    Trainer側のAPIに合わせて薄いラッパーを作成する.

    Example
    -------
    trainer = Trainer(model, trainer_cfg, train_loader, val_loader, callbacks)
    
    """
    trainer = Trainer(
        model=model,
        config=trainer_cfg,
        callbacks=callbacks,
        save_dir=save_dir
    )
    history = trainer.fit(train_loader, val_loader)
    return history


def main() -> None:
    start_time = time.time() # 学習開始時刻を記録
    print(">> Start training...")
    args = parse_args()
    cfg_path = Path(args.config)
    assert cfg_path.exists(), f"!! Config file {cfg_path} does not exist. !!"
    
    # 1. 基本設定
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # 必須セクションには[]でアクセス
    # オプションは.get()で取得, ない場合はデフォルト値を設定
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = cfg.get('save_dir', 'outputs')
    fix_seed(cfg.get('seed', 42), fix_cuda=True)  # シード固定

    # 2. DataLoader構築
    data_cfg = DataConfig(**cfg["data"]) # 必須セクション
    train_loader, val_loader = build_dataloaders(data_cfg)

    # 3. モデル構築
    model_cfg = cfg['model'] # 必須セクション
    model_class = getattr(models, model_cfg['name'])
    model = model_class(**model_cfg.get('params', {}))
    model.to(device)  # デバイスに移動

    # 4. Trainerの設定
    # trainer_cfgには全体の設定が含まれるためupdateする
    trainer_params = cfg.get('trainer', {})
    trainer_params.update({'device': device, 'save_dir': save_dir})
    trainer_cfg = TrainerConfig(**trainer_params)

    # 5. callbacksの動的設定
    callback_instances = []
    # callbacksセクションの設定をYAMLから読み込む
    callback_cfgs = cfg.get('callbacks', [])
    for cb_cfg in callback_cfgs:
        # 各callbackのクラス名とパラメータを取得
        cb_name = cb_cfg['name']
        cb_params = cb_cfg.get('params', {})
        # callbackクラスを取得してインスタンス化
        if hasattr(callback_module, cb_name):
            cb_class = getattr(callback_module, cb_name)
            callback_instances.append(cb_class(**cb_params))
        else:
            raise ValueError(f"!! Callback {cb_name} not found in callback_module. !!")
    # DefaultLoggerの追加
    logger = callback_module.DefaultLogger(save_dir=save_dir, log_interval=trainer_cfg.log_interval)
    callback_instances.append(logger)

    # 6. 学習実行
    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        trainer_cfg=trainer_cfg,
        callbacks=callback_instances,
        save_dir=save_dir
    )

    # 7. 終了処理
    elapsed_time = calc_elapsed_time(start_time)
    print(f"Training completed.")
    print(f"Elapsed time: {elapsed_time}")


if __name__ == "__main__":
    main()