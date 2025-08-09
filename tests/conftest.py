# -*- coding: utf-8 -*-
"""
Created on Aug 9, 2025

pytestの設定ファイル
- 隔離: 全てのテストはそれぞれ専用のクリーンな一時ディレクトリで実行される
- 自動化: 各テストコードに前準備のコードを書く必要がない
- 独立性: 各テストは独立して実行されるため, 他のテストの影響を受けない
- クリーニング: テスト終了後, 作成された一時ディレクトリやファイルは自動的に消去される

@author: tadahaya
"""

import os
import pytest
from pathlib import Path

@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    """
    各テストを一時ディレクトリで実行(outputsなどの副作用を隔離)

    """
    monkeypatch.chdir(tmp_path)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("data/processed/train").mkdir(parents=True, exist_ok=True)
    Path("data/processed/val").mkdir(parents=True, exist_ok=True)
    yield