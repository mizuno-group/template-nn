# -*- coding: utf-8 -*-
"""
Created on Friday August 15 15:23:40 2025

純汎用ユーティリティ.
重い責務(I/Oや可視化など)は扱わない.

@author: tadahaya
"""


from __future__ import annotations
import time
import random
from typing import Tuple, Callable

import numpy as np
import torch


# -*- coding: utf-8 -*-
"""
純汎用ユーティリティ（random seed の固定・経過時間の整形など）。
- ここでは I/O（ファイル保存）や可視化（matplotlib）など“重い責務”は扱いません。
- DataLoader 用の worker 初期化関数と torch.Generator を返すことで再現性を確保します。
- 旧実装との互換のため、fix_seed は (generator, seed_worker) を返します。
"""

from __future__ import annotations
import time
import random
from typing import Callable, Tuple

import numpy as np
import torch


def fix_seed(seed: int = 42, fix_cuda: bool = False) -> Tuple[torch.Generator, Callable[[int], None]]:
    """
    シードを固定し, DataLoaderで使えるgenerator/worker_init_fnを返す. 

    Parameters
    ----------
    seed : int, default=42
        
    fix_cuda : bool, default=False
        Trueのとき, cuDNNをdeterministicにしてアルゴリズム選択を固定
        再現性は高まるがしばしば遅くなる点に注意
        torchのバージョンによっては動作が異なる場合がある点にも注意

    Returns
    -------
    generator : torch.Generator
        DataLoaderに渡すためのCPU用Generator

    seed_worker : Callable[[int], None]
        各workerごとにシードを設定する初期化関数

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if fix_cuda:
            # 再現性重視
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # DataLoaderに渡すCPU側generator
    g = torch.Generator()
    g.manual_seed(seed)

    return g, seed_worker


def calc_elapsed_time(start_time: float) -> str:
    """
    経過時間を「Hh Mm S.sss」形式の文字列で返す.

    Parameters
    ----------
    start_time : float
        処理開始時刻, time.time()の戻り値

    Returns
    -------
    str
        "0h 00m 1.234s" のような文字列
    """
    elapsed = max(0.0, time.time() - float(start_time))
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed - (h * 3600 + m * 60)
    return f"{h}h {m}m {s:.3f}s"
