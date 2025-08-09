# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

データの前処理(クリーニング・正規化・特徴量生成など)をまとめる.
- Dataset本体の責務(入出力/IO)はdataset.pyへ, こちらは計算ロジックに集中.

@author: tadahaya
"""
from typing import Any

def identity(x: Any) -> Any:
    """何もしないサンプル(テンプレ差し込み用)"""
    return x
