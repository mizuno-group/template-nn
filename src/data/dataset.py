# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

データ周りの入り口(DatasetとDataLoaderをつくる役)
- PyTorchのDataset/DataLoaderを組み立てる責務に限定
- クリーニングや特徴量生成などの前処理はpreprocessing.py
- 画像解析を想定したDatasetクラスを実装

@author: tadahaya
"""
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image # 汎用の画像処理ライブラリ

# 1. Configコンテナ
@dataclass
class DataConfig:
    """
    データ読み込みの設定を管理するクラス.
    YAMLやJSONなどの設定ファイルから読み込んだdictをこの型に変換して使うことを想定.
    
    Attributes
    ----------

    """
    train_path: str
    val_path: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True


# 2. Datasetクラスの実装
class SimpleDataset(Dataset):
    """
    最小構成のカスタムDatasetクラス.
    画像解析を想定している.
    画像データとラベルを読み込み, 必要に応じて変換を適用する.
    IO律速になるように見えるが, DataLoaderのnum_workersを増やすことで
    並列処理が可能になるため, IOのボトルネックを軽減できる点が味噌.
     
    Parameters
    ----------
    data_path : str
        データセットのルートディレクトリのパス.
        各クラスの画像はこのディレクトリ内のサブディレクトリに格納されていることを想定.
        例: "data/train"

    transform : Optional[callable]
        画像に適用する変換関数.
        例えば, torchvision.transformsを使って画像の前処理を行うことができる.
        デフォルトはNoneで, その場合は画像をTensorに変換するだけの処理が行われる.
    
    """
    def __init__(
        self,
        data_path: str,
        transform:Optional[callable]=None,
        **kwargs: Any,
        ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.transform = transform

        # 1. クラス名とラベルIDの対応表を作成
        self.class_to_idx = {d.name: i for i, d in enumerate(self.data_path.iterdir()) if d.is_dir()}
        self.idx_to_class = {i: d for d, i in self.class_to_idx.items()}

        # 2. 全ての画像のパスとラベルIDのペアをリストに格納
        self.items = []
        for class_name, label_idx in self.class_to_idx.items():
            class_dir = self.data_path / class_name
            for image_path in class_dir.glob("*"):
                if image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.items.append((str(image_path), label_idx))


    def __len__(self) -> int:
        """
        データの総数を返す.
        ほとんど呼び出されないため, 効率はそれほど重要ではない.

        """
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        指定されたインデックスのデータサンプルを取得する.
        ラベルはint型で返すが, DataLoaderのcollate_fnのデフォルトで
        バッチ内のサンプルをまとめる際に, Tensor型に変換される.

        Parameters
        ----------
        idx : int
            Index of the data sample.

        """
        # 1. 指定されたidx番目のデータを取得
        image_path, label = self.items[idx]

        # 2. 画像の読み込み
        image = Image.open(image_path).convert("RGB")

        # 3. 画像の変換
        if self.transform:
            # 変換が指定されている場合はそれを適用, 内部にTensor変換が含まれていることを想定
            image = self.transform(image)
        else:
            # デフォルトの変換: Tensorに変換
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        return image, label


# 3. DataLoaderを組むための関数
def build_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """
    データローダーを構築する関数.
    
    Parameters
    ----------
    cfg : DataConfig
        データ読み込みの設定を含むDataConfigインスタンス.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        訓練用と検証用のDataLoaderのタプル.

    """
    # 1. Datasetのインスタンスを作成
    train_ds = SimpleDataset(data_path=cfg.train_path)
    
    # 2. DataLoaderを作成
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    
    # 3. 検証用データセットが指定されている場合は検証用DataLoaderも作成
    val_loader = None
    if cfg.val_path:
        val_ds = SimpleDataset(data_path=cfg.val_path, transform=None)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,  # 検証データはシャッフルしない
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
        )
    
    return train_loader, val_loader