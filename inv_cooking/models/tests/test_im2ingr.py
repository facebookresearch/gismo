import torch

from inv_cooking.models.im2ingr import Im2Ingr
from inv_cooking.models.ingredients_predictor.tests.utils import (
    FakeIngredientPredictorConfig,
)
from inv_cooking.models.tests.utils import FakeConfig


def test_Im2Ingr():
    torch.random.manual_seed(0)
    max_num_ingredients = 10
    vocab_size = 20
    batch_size = 5
    image = torch.randn(size=(batch_size, 3, 224, 224))
    labels = torch.randint(low=0, high=vocab_size - 1, size=(5, max_num_ingredients))

    ingredient_predictors = [
        (
            FakeIngredientPredictorConfig.ff_config(),
            torch.Size([batch_size, max_num_ingredients]),
        ),
        (
            FakeIngredientPredictorConfig.tf_config(),
            torch.Size([batch_size, max_num_ingredients + 1]),
        ),
        (
            FakeIngredientPredictorConfig.lstm_config(),
            torch.Size([batch_size, max_num_ingredients + 1]),
        ),
    ]

    for ingr_predictor, expected_prediction_shape in ingredient_predictors:
        model = Im2Ingr(
            image_encoder_config=FakeConfig.image_encoder_config(),
            ingr_pred_config=ingr_predictor,
            ingr_vocab_size=vocab_size,
            max_num_ingredients=max_num_ingredients,
            ingr_eos_value=vocab_size - 1,
        )

        losses, predictions = model(
            image, target_ingredients=labels, compute_losses=True, compute_predictions=True
        )

        assert losses["label_loss"] is not None
        assert predictions.shape == expected_prediction_shape
        assert predictions.min() >= 0
        assert predictions.max() <= vocab_size - 1
