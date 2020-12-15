import torch

from inv_cooking.models.im2ingr import Im2Ingr
from inv_cooking.models.tests.utils import FakeConfig


def test_Im2Ingr():
    torch.random.manual_seed(0)
    max_num_labels = 10
    vocab_size = 20
    batch_size = 5
    image = torch.randn(size=(batch_size, 3, 224, 224))
    labels = torch.randint(low=0, high=vocab_size - 1, size=(5, max_num_labels))

    ingredient_predictors = [
        (FakeConfig.ingr_pred_ff_config(), torch.Size([batch_size, max_num_labels])),
        (FakeConfig.ingr_pred_tf_config(), torch.Size([batch_size, max_num_labels + 1])),
        (FakeConfig.ingr_pred_lstm_config(), torch.Size([batch_size, max_num_labels + 1])),
    ]

    for ingr_predictor, expected_prediction_shape in ingredient_predictors:
        model = Im2Ingr(
            image_encoder_config=FakeConfig.image_encoder_config(),
            ingr_pred_config=ingr_predictor,
            ingr_vocab_size=vocab_size,
            dataset_name="recipe1m",
            max_num_labels=max_num_labels,
            ingr_eos_value=vocab_size - 1,
        )

        losses, predictions = model(
            image, label_target=labels, compute_losses=True, compute_predictions=True
        )

        assert losses["label_loss"] is not None
        assert predictions.shape == expected_prediction_shape
        assert predictions.min() >= 0
        assert predictions.max() <= vocab_size - 1
