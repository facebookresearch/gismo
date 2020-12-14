import torch
from omegaconf import DictConfig

from inv_cooking.config import ImageEncoderConfig, ImageEncoderFreezeType
from inv_cooking.models.im2ingr import Im2Ingr


def test_Im2Ingr():
    torch.random.manual_seed(0)
    max_num_labels = 10
    vocab_size = 20

    image_encoder_config = ImageEncoderConfig(
        model="resnet50",
        pretrained=False,
        dropout=0.1,
        freeze=ImageEncoderFreezeType.none,
    )
    ingr_pred_config = DictConfig(
        {
            "model": "ff_bce",
            "embed_size": 2048,
            "freeze": False,
            "load_pretrained_from": None,
            "layers": 2,
            "dropout": 0.0,
        }
    )
    model = Im2Ingr(
        image_encoder_config=image_encoder_config,
        ingr_pred_config=ingr_pred_config,
        ingr_vocab_size=vocab_size,
        dataset_name="recipe1m",
        max_num_labels=max_num_labels,
        ingr_eos_value=vocab_size - 1,
    )

    batch_size = 5
    image = torch.randn(size=(batch_size, 3, 224, 224))
    labels = torch.randint(low=0, high=vocab_size - 1, size=(5, max_num_labels))
    losses, predictions = model(
        image, label_target=labels, compute_losses=True, compute_predictions=True
    )

    assert losses["label_loss"] is not None
    assert predictions.shape == torch.Size([batch_size, max_num_labels])
    assert predictions.min() >= 0
    assert predictions.max() <= vocab_size - 1
