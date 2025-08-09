# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

評価のentry point
- YAMLファイルを読み込んで評価を開始する
- 評価の関数はTrainerクラスに定義

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
from src.trainer import Trainer
from src.utils.checkpoint import load_checkpoint
from src.utils.general import fix_seed, calc_elapsed_time


def parse_args() -> argparse.Namespace:
    """
    コマンドライン引数を解析する関数.
    
    Returns
    -------
    argparse.Namespace
        解析された引数を含むNamespaceオブジェクト.
    
    """
    parser = argparse.ArgumentParser(description="Evaluate entry")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, default="outputs/best.pt", help="Path to the model checkpoint file.")
    return parser.parse_args()


def run_evaluation(model, test_loader, device: str) -> Dict[str, Any]:
    """
    評価を実行するためのラッパー関数.
    Trainerクラスに評価メソッドがあると仮定.
    """
    # Trainerはモデルとデバイスのみで初期化（学習関連の設定は不要）
    trainer = Trainer(model=model, device=device)
    
    # Trainerの評価メソッドを呼び出す
    metrics = trainer.evaluate(test_loader)
    return metrics


def main() -> None:
    print(">> Start evaluation...")
    args = parse_args()
    cfg_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint_path)
    
    assert cfg_path.exists(), f"!! Config file {cfg_path} does not exist. !!"
    assert checkpoint_path.exists(), f"!! Checkpoint file {checkpoint_path} does not exist. !!"
    
    # 1. 基本設定
    with open(cfg_path, 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    fix_seed(cfg.get('seed', 42), fix_cuda=True)

    # 2. 評価用DataLoaderの構築
    # YAMLの'data'セクションに'test_path'を追加することを想定
    if "test_path" not in cfg["data"]:
        raise ValueError("!! 'test_path' not found in the data section of the config file. !!")
    
    # 評価なので、shuffleはFalseに固定
    data_cfg = DataConfig(test_path=cfg["data"]["test_path"], shuffle=False, **cfg["data"])
    _, test_loader = build_dataloaders(data_cfg) # build_dataloadersがtest_loaderを返すと仮定

    # 3. モデル構築
    model_cfg = cfg['model']
    model_class = getattr(models, model_cfg['name'])
    model = model_class(**model_cfg.get('params', {}))
    
    # ★★★ 4. 学習済み重みの読み込み ★★★
    # state_dict（モデルのパラメータ）をファイルからロード
    state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {checkpoint_path}")
    
    # モデルを評価モードに設定
    model.eval()
    model.to(device)

    # 5. 評価の実行
    metrics = run_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    # 6. 結果の表示
    print("\n>> Evaluation results:")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.4f}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()