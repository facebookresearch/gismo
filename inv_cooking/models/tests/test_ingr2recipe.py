# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from inv_cooking.models.ingr2recipe import Ingr2Recipe
from inv_cooking.models.tests.utils import FakeConfig


def test_Ingr2Recipe():
    torch.random.manual_seed(2)
    max_recipe_len = 15
    vocab_size = 10

    model = Ingr2Recipe(
        recipe_gen_config=FakeConfig.recipe_gen_config(),
        ingr_vocab_size=vocab_size,
        instr_vocab_size=vocab_size,
        max_recipe_len=max_recipe_len,
        ingr_eos_value=9,
    )

    batch_size = 5
    target_recipe = torch.randint(low=0, high=vocab_size, size=(batch_size, 10))
    ingredients = torch.randint(low=0, high=vocab_size, size=(batch_size, 10))

    losses, predictions = model(
        ingredients, target_recipe, compute_losses=True, compute_predictions=True
    )
    assert losses["recipe_loss"] is not None
    assert predictions is not None
    assert predictions.size(0) == batch_size
    assert predictions.size(1) == max_recipe_len
    assert predictions.min() >= 0
    assert predictions.max() <= vocab_size
