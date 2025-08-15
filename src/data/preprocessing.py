# -*- coding: utf-8 -*-
"""
Created on Friday August 15 15:22:54 2025

前処理ユーティリティ (I/Oを持たない純関数群)
- 画像/ndarray/PIL/Tensorを対象に安全にTransformを構成
- augmentは任意. 学習/評価で挙動を切り替えるbuild_transformsを提供

@author: tadahaya
"""
from __future__ import annotations
from typing import Optional, Tuple, Sequence, Callable, Dict, Any, List, Union


# torchvisionは関数内import (不要な際のimport時のI/Oを避けるため)
def _tv():
    import torchvision.transforms as T
    import torchvision.transforms.v2 as T2 # v2が利用可能ならより高機能
    try:
        # torchvision.transforms.v2 があればそちらを優先
        return T2
    except ImportError:
        return T


# 個別のTransform関数群
def to_tensor() -> Callable:
    return _tv().ToTensor()

def normalize(mean: Sequence[float], std: Sequence[float]) -> Callable:
    return _tv().Normalize(mean=mean, std=std)

def resize(size: Tuple[int, int]) -> Callable:
    return _tv().Resize(size)

def center_crop(size: Tuple[int, int]) -> Callable:
    return _tv().CenterCrop(size)

def random_resized_crop(size: Tuple[int, int]) -> Callable:
    return _tv().RandomResizedCrop(size)

def random_hflip(p: float = 0.5) -> Callable:
    return _tv().RandomHorizontalFlip(p=p)

def color_jitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) -> Callable:
    return _tv().ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

# 上記を組み合わせてTransformを構成する関数群
def compose_transforms(transforms: Sequence[Optional[Callable]]) -> Callable:
    """ Noneを除外して安全にCompose """
    return _tv().Compose([t for t in transforms if t is not None])


def build_transforms(
    *, 
    is_train: bool, 
    image_size: Tuple[int, int] = (224, 224),
    mean: Sequence[float] = (0.485, 0.456, 0.406), # ImageNet由来
    std: Sequence[float] = (0.229, 0.224, 0.225),
    use_augment: bool = True
) -> Callable:
    """
    学習/評価で使い分ける標準Transformセットを構築する.

        """
    transforms: List[Callable] = []

    if is_train:
        # --- 訓練時の変換 ---
        if use_augment:
            # データ拡張が有効な場合
            transforms.extend([
                random_resized_crop(image_size),
                random_hflip(),
                color_jitter()
            ])
        else:
            # データ拡張が無効な場合(デバッグ用など)
            transforms.extend([
                resize(image_size),
                center_crop(image_size)
            ])
    else:
        # --- 評価時の変換 (データ拡張なし) ---
        # 一般に一回り大きくリサイズしてから中央をクロップすると性能が良いとされる
        resize_size = int(image_size[0] * 256 / 224)
        transforms.extend([
            resize((resize_size, resize_size)),
            center_crop(image_size),
        ])

    # --- 共通の変換 ---
    # 最後にテンソル化と正規化を追加
    transforms.extend([
        to_tensor(),
        normalize(mean=mean, std=std),
    ])
    
    return compose_transforms(transforms)
