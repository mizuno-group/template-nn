# -*- coding: utf-8 -*-
"""
Created on Aug 9, 2025

@author: tadahaya
"""
import pytest
from src.utils.plot import progress_plot

def test_progress_plot(tmp_path ):
    # ダミーデータを作成
    epochs = list(range(1, 11))
    train_losses = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    val_losses = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.1, 0.05]
    
    # 一時ディレクトリに出力
    outdir = tmp_path / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    # プロットを実行
    progress_plot(
        outdir=str(outdir),
        epochs=epochs,
        train_losses=train_losses,
        val_losses=val_losses
    )
    # プロットファイルが作成されたか確認
    plot_file = outdir / "progress_loss.tif"
    assert plot_file.is_file()