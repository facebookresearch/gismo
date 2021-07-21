import pytest
import torch

from inv_cooking.config import ImageEncoderConfig
from inv_cooking.models.image_encoder.vit import create_vit_image_encoder


@pytest.mark.parametrize("patch_size,expected_seq_len", [[16, 576], [32, 144]])
def test_no_class_vit(patch_size: int, expected_seq_len: int):
    encoder = create_vit_image_encoder(
        embed_size=1024,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="vit",
            pretrained=False,
            freeze=False,
            n_cls_tokens=0,
            patch_size=patch_size,
        ),
        image_size=448,
    )
    x = torch.randn(size=(1, 3, 448, 448))
    y = encoder(x)
    assert y.shape == torch.Size([1, 1024, expected_seq_len])


@pytest.mark.parametrize("patch_size,pretrained", [[16, 32], [False, True]])
def test_one_class_vit_multi_layer_concatenated(patch_size: int, pretrained: bool):
    vit = create_vit_image_encoder(
        embed_size=1024,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="vit",
            pretrained=pretrained,
            freeze=False,
            n_cls_tokens=1,
            patch_size=patch_size,
            additional_repr_levels=[5, 8],
            concatenate_repr_levels=True,
        ),
        image_size=448,
    )
    x = torch.randn(size=(2, 3, 448, 448))
    out = vit(x)
    assert out.shape == torch.Size([2, 1024, 1])


@pytest.mark.parametrize("patch_size,pretrained", [[16, 32], [False, True]])
def test_one_class_vit_multi_layer_as_sequence(patch_size: int, pretrained: bool):
    vit = create_vit_image_encoder(
        embed_size=1024,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="vit",
            pretrained=pretrained,
            freeze=False,
            n_cls_tokens=1,
            patch_size=patch_size,
            additional_repr_levels=[5, 8],
            concatenate_repr_levels=False,
        ),
        image_size=448,
    )
    x = torch.randn(size=(2, 3, 448, 448))
    out = vit(x)
    assert out.shape == torch.Size([2, 1024, 3])


@pytest.mark.parametrize("patch_size,pretrained", [[16, 32], [False, True]])
def test_one_class_vit(patch_size: int, pretrained: bool):
    vit = create_vit_image_encoder(
        embed_size=1024,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="vit",
            pretrained=pretrained,
            freeze=False,
            n_cls_tokens=1,
            patch_size=patch_size,
        ),
        image_size=448,
    )

    x = torch.randn(size=(2, 3, 448, 448))
    out = vit(x)
    assert out.shape == torch.Size([2, 1024, 1])


@pytest.mark.parametrize("patch_size", [16, 32])
def test_multi_class_vit(patch_size: int):
    vit = create_vit_image_encoder(
        embed_size=1024,
        config=ImageEncoderConfig(
            dropout=0.5,
            model="vit",
            pretrained=False,
            freeze=False,
            n_cls_tokens=4,
            patch_size=patch_size,
        ),
        image_size=448,
    )
    x = torch.randn(size=(2, 3, 448, 448))
    out = vit(x)
    assert out.shape == torch.Size([2, 1024, 4])  # 5 to take into account the EOS token
