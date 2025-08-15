# -*- coding: utf-8 -*-
"""
Created on Friday August 15 15:39:49 2025

データセットとDataLoaderの構築
- DataConfig: 設定の入れ物
- build_dataloaders(dc): (train_loader, val_loader|None)を返す
- 画像ディレクトリ or numpy(.npy/.npz) の双方をサポート

@author: tadahaya
"""
# src/data/dataset.py

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

from src.utils.general import fix_seed
from . import preprocessing as PP

# ---- 1. Datasetの実装 ----
# src/data/dataset.py 内

class NpDataset(Dataset):
    """ メモリ上のNumPy配列を扱う. Tensorへの変換とCHW正規化を行う. """
    def __init__(self, data: np.ndarray, label: np.ndarray, transform: Optional[Callable] = None):
        assert data.ndim >= 3, "!! Data shape should be (N, C, H, W) or (N, H, W, C) !!"
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x, y = self.data[idx], self.label[idx]
        # NumPy配列をPyTorch Tensorに変換し, CHW形式に正規化する
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)  # (C, H, W) or (H, W, C)
            # チャンネルが先頭に来ていない場合(HWC形式), CHW形式に変換
            if x.ndim == 3 and x.shape[0] not in (1, 3):
                x = x.permute(2, 0, 1) # HWC -> CHW
            x = x.contiguous()

        # Transformを適用
        if self.transform:
            x = self.transform(x)
            
        return x, y

# ---- 2. DataConfig ----
@dataclass
class DataConfig:
    # 共通ローダー設定
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 42

    # データパス設定
    train_path: str
    val_path: Optional[str] = None
    
    # データ型 ('image_folder' | 'numpy')
    dataset_type: str = "image_folder"

    # 検証データの自動分割設定 (val_pathがない場合に使用)
    val_split: Optional[float] = None # 例: 0.2 (20%を検証用に)
    shuffle_split: bool = True # 分割時にシャッフルするか否か

    # Transformの設定
    transform_params: Dict[str, Any] = field(default_factory=dict)


# ---- 3. ヘルパー関数 ----
class _SubsetWithTransform(Dataset):
    """
    Subsetごとに異なるTransformを適用するためのラッパークラス.
    元のデータセットを共有しつつ, 個別の前処理を可能にする.

    """
    def __init__(self, dataset: Dataset, indices: list[int], transform: Optional[Callable]):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        # indicesを使って元のデータセットから, 未変換のデータを取得
        x, y = self.dataset[self.indices[idx]]
        # このサブセット専用のTransformを適用
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


def _create_dataset(
    path: str,
    dataset_type: str,
    transform: Optional[Callable] = None
) -> Dataset:
    """ パスと種類から単一のDatasetインスタンスを生成する. """
    if dataset_type == "image_folder":
        return ImageFolder(root=path, transform=transform)
    elif dataset_type == "numpy":
        X, Y = _load_numpy_pair(path)
        return NpDataset(X, Y, transform=transform)
    raise ValueError(f"!! 未対応の dataset_type: {dataset_type} !!")


def _split_dataset(
    train_ds: Dataset,
    val_split: float,
    seed: int,
    shuffle: bool,
    tfm_train: Callable, # 引数としてTransformを受け取る
    tfm_val: Callable    # 引数としてTransformを受け取る
) -> Tuple[Dataset, Dataset]:
    """
    訓練データセットをそれぞれ適切なTransformを持つ訓練用と検証用のサブセットに分割する.

    """
    n_samples = len(train_ds)
    indices = list(range(n_samples))
    split_idx = int(n_samples * (1 - val_split))
    
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # ラッパーに渡す前に, 元のデータセットのTransformを一時的に解除する.
    # ラッパーは常に生データを取得し, 独自のTransformを適用できる.
    original_transform = train_ds.transform
    train_ds.transform = None

    # _SubsetWithTransformを使って, それぞれにTransformを設定
    train_subset = _SubsetWithTransform(train_ds, train_indices, transform=tfm_train)
    val_subset = _SubsetWithTransform(train_ds, val_indices, transform=tfm_val)
    
    # 元のデータセットの状態を復元(丁寧な実装)
    train_ds.transform = original_transform
    
    return train_subset, val_subset


def _load_numpy_pair(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """ 単一の.npy/.npzファイルから(X, y)を読み込む. """
    data = np.load(path)
    # .npzの場合は 'x', 'y' というキーがあることを想定
    if isinstance(data, np.lib.npyio.NpzFile):
        return data['x'], data['y']
    raise TypeError(f"!! NumPyデータは'x'と'y'のkeyを持つ.npzファイル. !!")


# ---- 4. 公開API ----
def build_dataloaders(dc: DataConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    DataConfigに基づき, 訓練用と検証用のDataLoaderを構築する.

    """
    # Transformを構築
    tfm_train = PP.build_transforms(is_train=True, **dc.transform_params)
    tfm_val = PP.build_transforms(is_train=False, **dc.transform_params)
    
    # Datasetを構築
    train_ds = _create_dataset(dc.train_path, dc.dataset_type, tfm_train)
    val_ds = None

    if dc.val_path:
        val_ds = _create_dataset(dc.val_path, dc.dataset_type, tfm_val)
    elif dc.val_split:
        # _split_datasetにTransformを渡す
        train_ds, val_ds = _split_dataset(
            train_ds=train_ds,
            val_split=dc.val_split,
            seed=dc.seed,
            shuffle=dc.shuffle_split,
            tfm_train=tfm_train,
            tfm_val=tfm_val
        )

    # DataLoaderを生成
    g, seed_worker = fix_seed(dc.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=dc.batch_size,
        shuffle=True,
        num_workers=dc.num_workers,
        pin_memory=dc.pin_memory,
        generator=g,
        worker_init_fn=seed_worker
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds,
            batch_size=dc.batch_size * 2, # 検証では勾配計算しないので大きめに設定
            shuffle=False,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory,
            # 検証では乱数を使わないのでgenerator等は不要
        )

    return train_loader, val_loader