from typing import List

import pytest
import torch

from inv_cooking.config import ImageEncoderConfig
from inv_cooking.models.image_encoder import create_image_encoder
from inv_cooking.models.image_encoder.clip import ClipBasedEncoder
from inv_cooking.models.image_encoder.vit import VitImageEncoder


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


@pytest.mark.parametrize(
    "embed_size, type,expected", [
        (2048, "RN50", [1, 2048, 1]),
        (1024, "RN50", [1, 1024, 1]),
        (2048, "ViT-B/32", [1, 2048, 1]),
        (1024, "ViT-B/32", [1, 1024, 1])
    ]
)
def test_clip_encoder(embed_size: int, type: str, expected: List[int]):
    encoder = ClipBasedEncoder(
        embed_size=embed_size,
        config=ImageEncoderConfig(
            dropout=0.5,
            model=f"clip_{type}",
            pretrained=False,
            freeze=False,
        ),
    )
    x = torch.randn(size=(1, 3, 448, 448)).cuda()
    encoder.cuda(x.device)
    y = encoder(x)
    print(y.shape)
    assert y.shape == torch.Size(expected)


@pytest.mark.parametrize(
    "embed_size,expected", [(2048, [1, 2048, 49]), (1024, [1, 1024, 49])]
)
def test_vit_image_encoder(embed_size: int, expected: List[int]):
    encoder = VitImageEncoder(
        embed_size=embed_size,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="vit_32_small",
            pretrained=False,
            freeze=False,
        ),
    )
    x = torch.randn(size=(1, 3, 224, 224))
    y = encoder(x)
    assert y.shape == torch.Size(expected)


@pytest.mark.parametrize(
    "model_name", ["vit_32_small", "vit_16_small"]
)
def test_vit_image_encoder_size_448(model_name: str):
    encoder = create_image_encoder(
        embed_size=1024,
        config=ImageEncoderConfig(
            dropout=0.5,
            model=model_name,
            pretrained=False,
            freeze=False,
        ),
    )
    x = torch.randn(size=(1, 3, 448, 448))
    y = encoder(x)
    assert y.shape == torch.Size([1, 1024, 196])
