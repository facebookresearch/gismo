import torch

from inv_cooking.models.im2title import Im2Title
from inv_cooking.models.tests.utils import FakeConfig


def test_Im2Title():
    torch.random.manual_seed(2)
    ingr_vocab_size = instr_vocab_size = 20
    max_recipe_len = 15

    model = Im2Title(
        image_encoder_config=FakeConfig.image_encoder_config(),
        embed_size=2048,
        title_gen_config=FakeConfig.recipe_gen_config(),
        title_vocab_size=instr_vocab_size,
        max_title_len=max_recipe_len,
    )

    batch_size = 5
    image = torch.randn(size=(batch_size, 3, 224, 224))
    target_title = torch.randint(
        low=0, high=instr_vocab_size - 1, size=(batch_size, max_recipe_len)
    )

    losses, recipe_predictions = model(
        image,
        target_title,
        compute_losses=True,
        compute_predictions=True,
    )

    assert len(losses) >= 1
    assert losses["title_loss"] is not None
    assert recipe_predictions.shape == torch.Size([batch_size, max_recipe_len])
    assert recipe_predictions.max() <= ingr_vocab_size - 1
