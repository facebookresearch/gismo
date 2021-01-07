import torch

from inv_cooking.models.im2recipe import Im2Recipe
from inv_cooking.models.ingredients_predictor.tests.utils import (
    FakeIngredientPredictorConfig,
)
from inv_cooking.models.tests.utils import FakeConfig


def test_Im2Recipe():
    torch.random.manual_seed(2)
    ingr_vocab_size = instr_vocab_size = 20
    max_num_ingredients = 10
    max_recipe_len = 15

    model = Im2Recipe(
        image_encoder_config=FakeConfig.image_encoder_config(),
        ingr_pred_config=FakeIngredientPredictorConfig.ff_config(),
        recipe_gen_config=FakeConfig.recipe_gen_config(),
        ingr_vocab_size=ingr_vocab_size,
        instr_vocab_size=instr_vocab_size,
        max_num_ingredients=max_num_ingredients,
        max_recipe_len=max_recipe_len,
        ingr_eos_value=ingr_vocab_size - 1,
    )

    batch_size = 5
    image = torch.randn(size=(batch_size, 3, 224, 224))
    target_recipe = torch.randint(
        low=0, high=instr_vocab_size - 1, size=(batch_size, max_recipe_len)
    )
    ingr_gt = torch.randint(
        low=0, high=ingr_vocab_size - 1, size=(batch_size, max_num_ingredients)
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
    assert ingr_predictions.shape == torch.Size([batch_size, max_num_ingredients])
    assert recipe_predictions.shape == torch.Size([batch_size, max_recipe_len])
    assert ingr_predictions.max() <= ingr_vocab_size - 1
    assert recipe_predictions.max() <= ingr_vocab_size - 1
