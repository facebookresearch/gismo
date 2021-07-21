from typing import Any, Dict, Optional

import torch

from inv_cooking.config import (
    ImageEncoderConfig,
    IngredientPredictorConfig,
    IngredientTeacherForcingConfig,
    OptimizationConfig,
    PretrainedConfig,
    RecipeGeneratorConfig,
)
from inv_cooking.models.im2recipe import Im2Recipe
from inv_cooking.training.utils import MonitoredMetric, OptimizationGroup, _BaseModule
from inv_cooking.utils.metrics import (
    DistributedAverage,
    DistributedF1,
    DistributedValLosses,
)


class ImageToRecipe(_BaseModule):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        recipe_gen_config: RecipeGeneratorConfig,
        optim_config: OptimizationConfig,
        pretrained_im2ingr_config: PretrainedConfig,
        ingr_teachforce_config: IngredientTeacherForcingConfig,
        max_num_ingredients: int,
        max_recipe_len: int,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        ingr_eos_value: int,
    ):
        super().__init__()

        self.model = Im2Recipe(
            image_encoder_config,
            ingr_pred_config,
            recipe_gen_config,
            pretrained_im2ingr_config,
            ingr_vocab_size=ingr_vocab_size,
            instr_vocab_size=instr_vocab_size,
            max_num_ingredients=max_num_ingredients,
            max_recipe_len=max_recipe_len,
            ingr_eos_value=ingr_eos_value,
        )

        self.optimization = optim_config
        self.ingr_teachforce = ingr_teachforce_config

        # check if pretrained models
        self.pretrained_imenc = image_encoder_config.pretrained
        self.pretrained_ingrpred = (
            pretrained_im2ingr_config.load_pretrained_from != "None"
        )

        # metrics to track at validation time
        self.o_f1, self.c_f1, self.i_f1 = DistributedF1.create_all(
            which_f1=["o_f1", "c_f1", "i_f1"],
            pad_value=self.model.ingr_vocab_size - 1,
            remove_eos=self.model.ingr_predictor.requires_eos,
            dist_sync_on_step=True,
        )
        self.val_losses = DistributedValLosses(
            weights=self.optimization.loss_weights,
            monitor_ingr_losses=pretrained_im2ingr_config.freeze,
            dist_sync_on_step=True,
        )
        self.perplexity = DistributedAverage()

    def get_monitored_metric(self) -> MonitoredMetric:
        return MonitoredMetric(name="val_perplexity", mode="min")

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        ingredients: Optional[torch.Tensor] = None,
        recipe: Optional[torch.Tensor] = None,
        use_ingr_pred: bool = False,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ):
        out = self.model(
            image=image,
            target_recipe=recipe,
            target_ingredients=ingredients,
            use_ingr_pred=use_ingr_pred,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        return out[0], out[1:]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(
            compute_losses=True, use_ingr_pred=not self.ingr_teachforce.train, **batch
        )
        return out[0]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch, use_ingr_pred=not self.ingr_teachforce.val)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(batch, use_ingr_pred=not self.ingr_teachforce.test)

    def _evaluation_step(self, batch: Dict[str, torch.Tensor], use_ingr_pred: bool):
        out = self(
            **batch,
            use_ingr_pred=use_ingr_pred,
            compute_predictions=False,
            compute_losses=True,
        )
        out[0]["n_samples"] = batch["recipe"].shape[0]
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
        """
        Update distributed metrics
        """
        # compute ingredient metrics
        if step_output["ingr_pred"] is not None:
            self.o_f1(step_output["ingr_pred"], step_output["ingr_gt"])
            self.c_f1(step_output["ingr_pred"], step_output["ingr_gt"])
            self.i_f1(step_output["ingr_pred"], step_output["ingr_gt"])

        # update recipe metrics
        self.perplexity(torch.exp(step_output["recipe_loss"]))

        # update losses
        self.val_losses(step_output)

    def validation_epoch_end(self, out):
        self._eval_epoch_end(split="val")

    def test_epoch_end(self, out):
        self._eval_epoch_end(split="test")

    def _eval_epoch_end(self, split: str):
        if split == "test":
            self.log(f"{split}_o_f1", self.o_f1.compute())
            self.log(f"{split}_c_f1", self.c_f1.compute())
            self.log(f"{split}_i_f1", self.i_f1.compute())
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
                model=self.model.image_encoder,
                pretrained=self.pretrained_imenc,
                name="image encoder",
            ),
            OptimizationGroup(
                model=self.model.ingr_encoder,
                pretrained=False,
                name="ingredient encoder",
            ),
            OptimizationGroup(
                model=self.model.ingr_predictor,
                pretrained=self.pretrained_ingrpred,
                name="ingredient predictor",
            ),
            OptimizationGroup(
                model=self.model.recipe_gen,
                pretrained=False,
                name="recipe generator",
            ),
        ]
