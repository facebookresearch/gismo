import torch

from inv_cooking.models.ingredients_predictor import get_ingr_predictor
from .utils import FakeIngredientPredictorConfig


def test_ingredient_predictor():
    vocab_size = 20
    batch_size = 5
    max_num_labels = 10

    all_configs = [
        (FakeIngredientPredictorConfig.ff_config(), torch.Size([batch_size, max_num_labels])),
        (FakeIngredientPredictorConfig.lstm_config(), torch.Size([batch_size, max_num_labels + 1])),
        (FakeIngredientPredictorConfig.tf_config(), torch.Size([batch_size, max_num_labels + 1])),
    ]

    for config, expected_output_shape in all_configs:
        model = get_ingr_predictor(
            config,
            vocab_size=vocab_size,
            maxnumlabels=max_num_labels,
            eos_value=vocab_size - 1,
        )

        image_features = torch.randn(size=(batch_size, config.embed_size, 49))
        label_target = torch.randint(
            low=0, high=vocab_size - 1, size=(batch_size, max_num_labels)
        )
        losses, predictions = model(
            image_features, label_target, compute_losses=True, compute_predictions=True,
        )

        assert losses["label_loss"] is not None
        assert predictions.shape == expected_output_shape
        assert predictions.min() >= 0
        assert predictions.max() < vocab_size
