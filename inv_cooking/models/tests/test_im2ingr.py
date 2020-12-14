import torch

from inv_cooking.models.im2ingr import Im2Ingr
from inv_cooking.models.tests.utils import FakeConfig


def test_Im2Ingr():
    torch.random.manual_seed(0)
    max_num_labels = 10
    vocab_size = 20

    model = Im2Ingr(
        image_encoder_config=FakeConfig.image_encoder_config(),
        ingr_pred_config=FakeConfig.ingr_pred_config(),
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
