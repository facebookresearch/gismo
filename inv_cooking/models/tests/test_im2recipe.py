import torch

from inv_cooking.models.im2recipe import Im2Recipe
from inv_cooking.models.ingredients_predictor.tests.utils import (
    FakeIngredientPredictorConfig,
)
from inv_cooking.models.tests.utils import FakeConfig


def test_Im2Recipe():
    torch.random.manual_seed(2)
    vocab_size = 20
    max_num_labels = 10
    max_recipe_len = 15

    model = Im2Recipe(
        image_encoder_config=FakeConfig.image_encoder_config(),
        ingr_pred_config=FakeIngredientPredictorConfig.ff_config(),
        recipe_gen_config=FakeConfig.recipe_gen_config(),
        ingr_vocab_size=vocab_size,
        instr_vocab_size=vocab_size,
        max_num_labels=max_num_labels,
        max_recipe_len=max_recipe_len,
        ingr_eos_value=vocab_size - 1,
    )

    batch_size = 5
    image = torch.randn(size=(batch_size, 3, 224, 224))
    target_recipe = torch.randint(
        low=0, high=vocab_size - 1, size=(batch_size, max_recipe_len)
    )
    ingr_gt = torch.randint(
        low=0, high=vocab_size - 1, size=(batch_size, max_num_labels)
    )

    losses, ingr_predictions, recipe_predictions = model(
        image,
        target_recipe,
        ingr_gt,
        use_ingr_pred=True,
        compute_losses=True,
        compute_predictions=True,
    )

    assert len(losses) >= 1
    assert losses["recipe_loss"] is not None
    assert ingr_predictions.shape == torch.Size([batch_size, max_num_labels])
    assert recipe_predictions.shape == torch.Size([batch_size, max_recipe_len])
    assert ingr_predictions.max() <= vocab_size - 1
    assert recipe_predictions.max() <= vocab_size - 1
