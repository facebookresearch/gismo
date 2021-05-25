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

        self.model = Im2Title(
            image_encoder_config=image_encoder_config,
            embed_size=ingr_pred_config.embed_size,
            title_gen_config=title_gen_config,
            title_vocab_size=title_vocab_size,
            max_title_len=max_title_len,
        )

        self.optimization = optim_config

        # check if pretrained models
        self.pretrained_imenc = image_encoder_config.pretrained

        # metrics to track at validation time
        self.perplexity = DistributedAverage()

    def get_monitored_metric(self) -> MonitoredMetric:
        return MonitoredMetric(name="val_perplexity", mode="min")

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        title: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ):
        assert isinstance(self.model, Im2Title)
        out = self.model(
            image=image,
            target_title=title,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        return out[0], out[1:]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(compute_losses=True, **batch)
        return out[0]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch)

    def _evaluation_step(self, batch: Dict[str, torch.Tensor]):
        out = self(**batch, compute_predictions=False, compute_losses=True)
        out[0]["n_samples"] = batch["title"].shape[0]
        return out[0]

    def training_step_end(self, losses: Dict[str, torch.Tensor]):
        return self.log_training_losses(losses, self.optimization)

    def validation_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def test_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def _evaluation_step_end(self, step_output: Dict[str, Any]):
        self.perplexity(torch.exp(step_output["recipe_loss"]))

    def validation_epoch_end(self, out):
        self._eval_epoch_end(split="val")

    def test_epoch_end(self, out):
        self._eval_epoch_end(split="test")

    def _eval_epoch_end(self, split: str):
        self.log(f"{split}_perplexity", self.perplexity.compute())

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
                model=self.model.img_features_transform,
                pretrained=False,
                name="image feature transform",
            ),
            OptimizationGroup(
                model=self.model.recipe_gen,
                pretrained=False,
                name="recipe generator",
            ),
        ]
