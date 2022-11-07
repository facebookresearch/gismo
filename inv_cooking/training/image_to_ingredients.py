# Copyright (c) Meta Platforms, Inc. All Rights Reserved
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
)
from inv_cooking.config.config import TitleEncoderConfig
from inv_cooking.models.im2ingr import Im2Ingr
from inv_cooking.training.utils import MonitoredMetric, OptimizationGroup, _BaseModule
from inv_cooking.utils.metrics import DistributedF1, DistributedValLosses


class ImageToIngredients(_BaseModule):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        title_encoder_config: TitleEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        optim_config: OptimizationConfig,
        max_num_ingredients: int,
        title_vocab_size: int,
        ingr_vocab_size: int,
        ingr_eos_value: int,
    ):
        super().__init__()
        self.model = Im2Ingr(
            image_encoder_config,
            title_encoder_config,
            ingr_pred_config,
            title_vocab_size=title_vocab_size,
            ingr_vocab_size=ingr_vocab_size,
            max_num_ingredients=max_num_ingredients,
            ingr_eos_value=ingr_eos_value,
        )

        self.optimization = optim_config

        # check if pre-trained image encoder
        self.pretrained_imenc = image_encoder_config.pretrained

        # metrics to track at validation time
        self.o_f1, self.c_f1, self.i_f1 = DistributedF1.create_all(
            which_f1=["o_f1", "c_f1", "i_f1"],
            pad_value=self.model.ingr_vocab_size - 1,
            remove_eos=self.model.ingr_predictor.requires_eos,
            dist_sync_on_step=True,
        )
        self.val_losses = DistributedValLosses(
            weights=self.optimization.loss_weights,
            monitor_ingr_losses=True,
            dist_sync_on_step=True,
        )

    def get_monitored_metric(self) -> MonitoredMetric:
        return MonitoredMetric(name="val_o_f1", mode="max")

    def forward(
        self,
        image: Optional[torch.Tensor],
        title: Optional[torch.Tensor] = None,
        ingredients: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ):
        out = self.model(
            image=image,
            title=title,
            target_ingredients=ingredients,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        return out[0], out[1:]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(
            image=batch.get("image", None),
            title=batch.get("title", None),
            ingredients=batch["ingredients"],
            compute_losses=True,
        )
        return out[0]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch)

    def _evaluation_step(self, batch: Dict[str, torch.Tensor]):
        out = self(
            image=batch.get("image", None),
            title=batch.get("title", None),
            ingredients=batch["ingredients"],
            compute_predictions=True,
            compute_losses=True,
        )
        out[0]["n_samples"] = batch["ingredients"].shape[0]
        out[0]["ingr_pred"] = out[1][0]
        out[0]["ingr_gt"] = batch["ingredients"]
        return out[0]

    def training_step_end(self, losses: Dict[str, torch.Tensor]):
        return self.log_training_losses(losses, self.optimization)

    def validation_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def test_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def _evaluation_step_end(self, step_output: Dict[str, Any]):
        self.o_f1(step_output["ingr_pred"], step_output["ingr_gt"])
        self.c_f1(step_output["ingr_pred"], step_output["ingr_gt"])
        self.i_f1(step_output["ingr_pred"], step_output["ingr_gt"])
        self.val_losses(step_output)

    def validation_epoch_end(self, out):
        self._eval_epoch_end(split="val")

    def test_epoch_end(self, out):
        self._eval_epoch_end(split="test")

    def _eval_epoch_end(self, split: str):
        self.log(f"{split}_o_f1", self.o_f1.compute())
        self.log(f"{split}_c_f1", self.c_f1.compute())
        self.log(f"{split}_i_f1", self.i_f1.compute())
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
                model=self.model.image_encoder,
                pretrained=self.pretrained_imenc,
                name="image encoder",
            ),
            OptimizationGroup(
                model=self.model.ingr_predictor,
                pretrained=False,
                name="ingredient predictor",
            ),
        ]
