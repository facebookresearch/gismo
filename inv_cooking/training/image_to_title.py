# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch

from inv_cooking.config import (
    ImageEncoderConfig,
    IngredientPredictorConfig,
    OptimizationConfig,
    RecipeGeneratorConfig,
)
from inv_cooking.models.im2title import Im2Title
from inv_cooking.training.utils import MonitoredMetric, OptimizationGroup, _BaseModule
from inv_cooking.utils.metrics import DistributedAverage


class ImageToTitle(_BaseModule):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        title_gen_config: RecipeGeneratorConfig,
        optim_config: OptimizationConfig,
        max_title_len: int,
        title_vocab_size: int,
    ):
        super().__init__()
        self.optim_config = optim_config
        self.model = Im2Title(
            image_encoder_config=image_encoder_config,
            embed_size=ingr_pred_config.embed_size,
            title_gen_config=title_gen_config,
            title_vocab_size=title_vocab_size,
            max_title_len=max_title_len,
        )

        # check if pretrained models
        self.pretrained_imenc = image_encoder_config.pretrained

        # metrics to track at validation time
        self.val_title_loss = DistributedAverage()
        self.val_perplexity = DistributedAverage()

    def get_monitored_metric(self) -> MonitoredMetric:
        return MonitoredMetric(name="val_title_loss", mode="min")

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        title: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ):
        out = self.model(
            image=image,
            target_title=title,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        return out[0], out[1:]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(
            image=batch["image"],
            title=batch["title"],
            compute_losses=True,
            compute_predictions=False,
        )
        return out[0]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch)

    def _evaluation_step(self, batch: Dict[str, torch.Tensor]):
        out = self(
            image=batch["image"],
            title=batch["title"],
            compute_losses=True,
            compute_predictions=False,
        )
        return out[0]

    def training_step_end(self, losses: Dict[str, torch.Tensor]):
        return self.log_training_losses(losses, self.optim_config)

    def validation_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def test_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def _evaluation_step_end(self, step_output: Dict[str, Any]):
        self.val_title_loss(step_output["title_loss"])
        self.val_perplexity(torch.exp(step_output["title_loss"]))

    def validation_epoch_end(self, out):
        self._eval_epoch_end(split="val")

    def test_epoch_end(self, out):
        self._eval_epoch_end(split="test")

    def _eval_epoch_end(self, split: str):
        self.log(f"{split}_title_loss", self.val_title_loss.compute())
        self.log(f"{split}_title_perplexity", self.val_perplexity.compute())

    def configure_optimizers(self):
        return self.create_optimizers(
            optim_groups=self.create_optimization_groups(),
            optim_config=self.optim_config,
        )

    def create_optimization_groups(self):
        return [
            OptimizationGroup(
                model=self.model.image_encoder,
                pretrained=self.pretrained_imenc,
                name="image encoder",
            ),
            OptimizationGroup(
                model=self.model.img_features_transform,
                pretrained=False,
                name="image feature transform",
            ),
            OptimizationGroup(
                model=self.model.recipe_gen,
                pretrained=False,
                name="title generator",
            ),
        ]
