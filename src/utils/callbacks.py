# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

学習中に各エポック／各イテレーションで呼ばれる処理(=callback)をまとめる.
- EarlyStopping
- DefaultLogger

@author: tadahaya
"""
import json, os, time, yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import torch


class EarlyStopping:
    def __init__(self, patience=10, mode="min", restore_best_model=True, verbose=True):
        """
        Early stopping

        Parameters
        ----------
        patience: int
            number of epochs to wait before stopping

        mode: str
            "min" or "max" (loss or accuracy)

        restore_best_model: bool
            whether to restore the best model

        verbose: bool
            whether to print messages

        """
        self.patience = patience
        self.restore_best_model = restore_best_model
        self.verbose = verbose
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        self._monitor_fxn = {
            "min": lambda a, b: a < b,
            "max": lambda a, b: a > b
        }[mode]

    def __call__(self, model, score, epoch):
        """
        Parameters
        ----------
        model: torch.nn.Module
            current model

        score: float
            current score (loss or accuracy)

        epoch: int
            current epoch

        """
        if self.best_score is None or self._monitor_fxn(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_model:
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                # store the best model state on CPU
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(">> EarlyStopping triggered")
                if self.restore_best_model and self.best_model_state:
                    model.load_state_dict(self.best_model_state)


@dataclass
class DefaultLogger:
    """
    学習の進捗を簡易に記録するロガー。
    - コンソール出力（print）
    - JSON で履歴をファイル保存（任意）
    """
    save_dir: str = "outputs"
    log_interval: int = 50
    history: Dict[str, List[float]] = field(default_factory=lambda: {})
    _t0: float = field(default_factory=time.time)

    def log_iter(self, step: int, metrics: Dict[str, float]) -> None:
        """学習ループ内（バッチごと）で一定間隔ごとに呼ぶ想定。"""
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(float(v))
        if step % self.log_interval == 0:
            msg = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"[iter {step}] {msg}")

    def log_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """エポック終了時にまとめて呼ぶ想定。"""
        for k, v in metrics.items():
            self.history.setdefault(k, []).append(float(v))
        msg = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        elapsed = time.time() - self._t0
        print(f"[epoch {epoch}] {msg} | elapsed: {elapsed:.1f}s")

    def save(self, filename: str = "logs.json") -> None:
        """履歴を JSON で保存（必要なときだけ呼ぶ）。"""
        out = Path(self.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / filename, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)