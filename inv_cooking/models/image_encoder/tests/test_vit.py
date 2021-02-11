from typing import List

import pytest
import torch

from inv_cooking.config import ImageEncoderConfig
from inv_cooking.models.image_encoder.vit import OneClassVit, NoClassVit


def test_one_class_vit():
    vit = OneClassVit(
        embed_size=1024,
        config=ImageEncoderConfig(pretrained=False, dropout=0.1),
        image_size=448
    )

    x = torch.randn(size=(2, 3, 448, 448))
    out = vit(x)
    assert out.shape == torch.Size([2, 1024, 1])


@pytest.mark.parametrize(
    "model_name", ["vit_32_small", "vit_16_small"]
)
def test_vit_image_encoder_size_448(model_name: str):
    encoder = NoClassVit(
        embed_size=1024,
        config=ImageEncoderConfig(
            dropout=0.5,
            model=model_name,
            pretrained=False,
            freeze=False,
        ),
        image_size=448
    )
    x = torch.randn(size=(1, 3, 448, 448))
    y = encoder(x)
    assert y.shape == torch.Size([1, 1024, 196])
