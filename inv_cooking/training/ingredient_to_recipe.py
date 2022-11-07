# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch

from inv_cooking.config import OptimizationConfig, RecipeGeneratorConfig
from inv_cooking.models.ingr2recipe import Ingr2Recipe
from inv_cooking.training.utils import MonitoredMetric, OptimizationGroup, _BaseModule
from inv_cooking.utils.metrics import DistributedAverage, DistributedValLosses


class IngredientToRecipe(_BaseModule):
    def __init__(
        self,
        recipe_gen_config: RecipeGeneratorConfig,
        optim_config: OptimizationConfig,
        max_recipe_len: int,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        ingr_eos_value: int,
    ):
        super().__init__()

        self.model = Ingr2Recipe(
            recipe_gen_config,
            ingr_vocab_size,
            instr_vocab_size,
            max_recipe_len=max_recipe_len,
            ingr_eos_value=ingr_eos_value,
        )

        self.optimization = optim_config

        # metrics to track at validation time
        self.val_losses = DistributedValLosses(
            weights=self.optimization.loss_weights, dist_sync_on_step=True
        )
        self.perplexity = DistributedAverage()

    def get_monitored_metric(self) -> MonitoredMetric:
        return MonitoredMetric(name="val_perplexity", mode="min")

    def forward(
        self,
        ingredients: Optional[torch.Tensor] = None,
        recipe: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ):
        out = self.model(
            ingredients=ingredients,
            target_recipe=recipe,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        return out[0], out[1:]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(
            compute_losses=True,
            ingredients=batch["ingredients"],
            recipe=batch["recipe"],
        )
        return out[0]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch)

    def _evaluation_step(self, batch: Dict[str, torch.Tensor]):
        out = self(
            ingredients=batch["ingredients"],
            recipe=batch["recipe"],
            compute_predictions=False,
            compute_losses=True,
        )
        out[0]["n_samples"] = batch["recipe"].shape[0]
        return out[0]

    def training_step_end(self, losses: Dict[str, torch.Tensor]):
        return self.log_training_losses(losses, self.optimization)

    def validation_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def test_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def _evaluation_step_end(self, step_output: Dict[str, Any]):
        self.perplexity(torch.exp(step_output["recipe_loss"]))
        self.val_losses(step_output)

    def validation_epoch_end(self, out):
        self._eval_epoch_end(split="val")

    def test_epoch_end(self, out):
        self._eval_epoch_end(split="test")

    def _eval_epoch_end(self, split: str):
        self.log(f"{split}_perplexity", self.perplexity.compute())
        val_losses = self.val_losses.compute()
        for k, v in val_losses.items():
            self.log(f"{split}_{k}", v)

    def configure_optimizers(self):
        return self.create_optimizers(
            optim_groups=self.create_optimization_groups(),
            optim_config=self.optimization,
        )

    def create_optimization_groups(self):
        return [
            OptimizationGroup(
                model=self.model.ingr_encoder,
                name="ingredient encoder",
                pretrained=False,
            ),
            OptimizationGroup(
                model=self.model.recipe_gen,
                name="recipe generator",
                pretrained=False,
            ),
        ]
