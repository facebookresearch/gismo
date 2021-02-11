from typing import List

import pytest
import torch

from inv_cooking.config import ImageEncoderConfig
from inv_cooking.models.image_encoder import create_image_encoder


@pytest.mark.parametrize(
    "embed_size,expected", [(2048, [1, 2048, 49]), (1024, [1, 1024, 49])]
)
def test_resnet_image_encoder(embed_size: int, expected: List[int]):
    encoder = create_image_encoder(
        embed_size=embed_size,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="resnet50",
            pretrained=True,
            freeze=False,
        ),
    )
    x = torch.randn(size=(1, 3, 224, 224))
    y = encoder(x)
    assert y.shape == torch.Size(expected)


def test_resnet_image_encoder_size_448():
    encoder = create_image_encoder(
        embed_size=1024,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="resnet50",
            pretrained=True,
            freeze=False,
        ),
    )
    x = torch.randn(size=(1, 3, 448, 448))
    y = encoder(x)
    assert y.shape == torch.Size([1, 1024, 196])
