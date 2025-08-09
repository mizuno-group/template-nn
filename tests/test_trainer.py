# -*- coding: utf-8 -*-
"""

モデル初期化や推論のテスト例

@author: tadahaya
"""
import torch
from src.models.model import create_model

def test_model_initialization():
    model = create_model("custom_cnn", num_classes=10)
    sample_input = torch.randn(1, 3, 224, 224)
    output = model(sample_input)
    assert output.shape == (1, 10), "Model output shape should match (1, num_classes)"