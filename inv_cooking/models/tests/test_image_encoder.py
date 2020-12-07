from typing import List

import pytest
import torch

from inv_cooking.config import ImageEncoderConfig, ImageEncoderFreezeType
from inv_cooking.models.image_encoder import ImageEncoder


@pytest.mark.parametrize(
    "embed_size,expected", [(2048, [1, 2048, 49]), (1024, [1, 1024, 49])]
)
def test_image_encoder(embed_size: int, expected: List[int]):
    encoder = ImageEncoder(
        embed_size=embed_size,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="resnet50",
            pretrained=True,
            freeze=ImageEncoderFreezeType.none
        )
    )
    x = torch.randn(size=(1, 3, 224, 224))
    y = encoder(x)
    assert y.shape == torch.Size(expected)