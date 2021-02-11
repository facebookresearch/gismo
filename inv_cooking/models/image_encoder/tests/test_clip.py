from typing import List

import pytest
import torch

from inv_cooking.config import ImageEncoderConfig
from inv_cooking.models.image_encoder.clip import ClipBasedEncoder


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
