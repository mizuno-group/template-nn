# -*- coding: utf-8 -*-
"""
Created on Friday August 15 15:24:05 2025

checkpointや学習履歴を保存・読み込みする関数をまとめる (純I/O). 

@author: tadahaya
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional
import json
import yaml

import torch

class ExperimentSnapshot(NamedTuple):
    config: Dict[str, Any]
    history: Dict[str, Any]
    model_state: Optional[Dict[str, Any]]
    optimizer_state: Optional[Dict[str, Any]]
    scheduler_state: Optional[Dict[str, Any]]


def save_snapshot(
    *, # キーワード引数を強制
    outdir: str | Path,
    config: Dict[str, Any],
    history: Dict[str, Any],
    model: Any, # torch.nn.Module
    optimizer: Any, # torch.optim.Optimizer
    scheduler: Optional[Any] = None,
    name: str = "final",
) -> None:
    """
    実験スナップショット一式(設定, 履歴, 状態辞書)を保存する.
    
    outdir/
      ├─ config.yaml
      ├─ history.json
      └─ model_{name}.pt
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # config
    (outdir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    # history
    (outdir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # states
    cp_path = outdir / f"model_{name}.pt"
    pkg = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        try:
            pkg["scheduler"] = scheduler.state_dict()
        except AttributeError:
            # .state_dict()を持たないスケジューラもあるため
            pass
    torch.save(pkg, cp_path)
    print(f"Saved snapshot to {outdir}")


def load_snapshot(
    *, # キーワード引数を強制
    resdir: str | Path,
    checkpoint_name: str = "model_final",
) -> ExperimentSnapshot:
    """
    実験スナップショット(設定, 履歴, 状態辞書)をファイルから読み込む.
    ファイルの読み込みに専念し, 状態の適用は行わない.
    
    Returns
    -------
    ExperimentSnapshot
        実験の状態をまとめた名前付きタプル.

    """
    resdir = Path(resdir)

    # config / history の読み込み
    config_path = resdir / "config.yaml"
    history_path = resdir / "history.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}

    # states (モデル等の状態辞書)の読み込み
    ckpt_path = _normalize_ckpt_name(resdir, checkpoint_name)
    model_state, opt_state, sch_state = _load_states(ckpt_path) if ckpt_path.exists() else (None, None, None)

    return ExperimentSnapshot(
        config=config,
        history=history,
        model_state=model_state,
        optimizer_state=opt_state,
        scheduler_state=sch_state
    )


def _load_states(path: Path) -> tuple[Dict, Optional[Dict], Optional[Dict]]:
    """1つの.ptファイルからstate_dict群を辞書で受け取る."""
    pkg = torch.load(path, map_location="cpu")
    return pkg.get("model", {}), pkg.get("optimizer"), pkg.get("scheduler")


def _normalize_ckpt_name(resdir: Path, name: str) -> Path:
    """ "model_final" のような拡張子なし指定でも安全に .pt を補う."""
    p = resdir / name
    return p.with_suffix(".pt") if p.suffix != ".pt" else p