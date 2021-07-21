import torch

from inv_cooking.training.ingredient_to_recipe import IngredientToRecipe
from inv_cooking.training.tests.utils import _BaseTest


class TestIngredientToRecipe(_BaseTest):
    @torch.no_grad()
    def test_ingr2recipe(self):
        module = IngredientToRecipe(
            recipe_gen_config=self.default_recipe_generator_config(),
            optim_config=self.default_optimization_config(),
            max_recipe_len=self.MAX_NUM_INSTRUCTIONS * self.MAX_INSTRUCTION_LENGTH,
            ingr_vocab_size=self.INGR_VOCAB_SIZE,
            instr_vocab_size=self.RECIPE_VOCAB_SIZE,
            ingr_eos_value=self.INGR_EOS_VALUE,
        )

        batch_size = 5
        ingredients = torch.randint(
            low=0,
            high=self.INGR_VOCAB_SIZE,
            size=(batch_size, self.MAX_NUM_INGREDIENTS + 1),
        )
        recipe = torch.randint(
            low=0, high=self.RECIPE_VOCAB_SIZE, size=(batch_size, self.MAX_RECIPE_LEN)
        )

        # Try building an optimizer
        self.assert_all_parameters_used(module)
        optimizers, schedulers = module.configure_optimizers()
        assert len(optimizers) == 1
        assert len(schedulers) == 1

        # Try "train" forward pass
        losses, predictions = module(
            ingredients=ingredients,
            recipe=recipe,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 1
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert len(predictions) == 1
        assert predictions[0].shape == torch.Size([5, self.MAX_RECIPE_LEN])

        # Try "train" step
        batch = dict(ingredients=ingredients, recipe=recipe)
        losses = module.training_step(batch=batch, batch_idx=0)
        assert len(losses) == 1
        assert losses["recipe_loss"].shape == torch.Size([5])

        # Try "val" step
        losses = module.validation_step(batch=batch, batch_idx=0)
        assert len(losses) == 2
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert losses["n_samples"] == 5

        # Try "test" step
        losses = module.test_step(batch=batch, batch_idx=0)
        assert len(losses) == 2
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert losses["n_samples"] == 5
