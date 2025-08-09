# -*- coding: utf-8 -*-
"""

データ処理関係のテスト例

@author: tadahaya
"""
import pytest
import pandas as pd
from src.data.prepare import prepare_data

@pytest.fixture
def sample_config():
    return {
        "raw_data_dir": "data/raw/",
        "processed_data_dir": "data/processed/",
        "train_data": "data/processed/train.csv"
    }

def test_prepare_data(sample_config):
    prepare_data(sample_config)
    df = pd.read_csv(sample_config["train_data"])
    assert not df.empty, "Processed train data should not be empty"
