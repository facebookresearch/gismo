# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from inv_cooking.config.config import TitleEncoderConfig
from inv_cooking.models.title_encoder import TitleEncoder


@pytest.mark.parametrize("layers", [0, 2])
@pytest.mark.parametrize("layer_dim", [512, 2048])
def test_title_encoder(layers: int, layer_dim: int):
    encoder = TitleEncoder(
        config=TitleEncoderConfig(with_title=True, layer_dim=layer_dim, layers=layers),
        title_vocab_size=100,
        embed_size=2048,
    )
    x = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = encoder(x)
    assert y.shape == torch.Size([2, 2048, 4])
