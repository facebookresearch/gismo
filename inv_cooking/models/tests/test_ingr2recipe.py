import torch

from inv_cooking.config import RecipeGeneratorConfig
from inv_cooking.models.ingr2recipe import Ingr2Recipe


def test_Ingr2Recipe():
    torch.random.manual_seed(2)
    max_recipe_len = 15
    vocab_size = 10

    recipe_generator_config = RecipeGeneratorConfig(
        dropout=0.1, embed_size=3, n_att_heads=1, layers=1, normalize_before=True
    )
    model = Ingr2Recipe(
        recipe_gen_config=recipe_generator_config,
        ingr_vocab_size=vocab_size,
        instr_vocab_size=vocab_size,
        max_recipe_len=max_recipe_len,
        ingr_eos_value=9,
    )

    batch_size = 5
    recipe_target = torch.randint(low=0, high=vocab_size, size=(batch_size, 10))
    ingredients = torch.randint(low=0, high=vocab_size, size=(batch_size, 10))

    losses, predictions = model(
        recipe_target, ingredients, compute_losses=True, compute_predictions=True
    )
    assert losses["recipe_loss"] is not None
    assert predictions is not None
    assert predictions.size(0) == batch_size
    assert predictions.size(1) == max_recipe_len
    assert predictions.min() >= 0
    assert predictions.max() <= vocab_size
