from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ExponentialLR

from inv_cooking.config import (
    ImageEncoderConfig,
    IngredientPredictorConfig,
    OptimizationConfig,
    RecipeGeneratorConfig,
    TaskType,
)
from inv_cooking.models.im2ingr import Im2Ingr
from inv_cooking.models.im2recipe import Im2Recipe
from inv_cooking.models.ingr2recipe import Ingr2Recipe
from inv_cooking.utils.metrics import (
    DistributedF1,
    DistributedMetric,
    DistributedValLosses,
)


class LitInverseCooking(pl.LightningModule):
    def __init__(
        self,
        task: TaskType,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        recipe_gen_config: RecipeGeneratorConfig,
        optim_config: OptimizationConfig,
        max_num_labels: int,
        max_recipe_len: int,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        ingr_eos_value: int,
    ):
        super().__init__()

        if task == TaskType.im2ingr:
            self.model = Im2Ingr(
                image_encoder_config,
                ingr_pred_config,
                ingr_vocab_size,
                max_num_labels,
                ingr_eos_value,
            )
        elif task == TaskType.im2recipe:
            self.model = Im2Recipe(
                image_encoder_config,
                ingr_pred_config,
                recipe_gen_config,
                ingr_vocab_size,
                instr_vocab_size,
                max_num_labels,
                max_recipe_len,
                ingr_eos_value,
            )
        elif task == TaskType.ingr2recipe:
            self.model = Ingr2Recipe(
                recipe_gen_config,
                ingr_vocab_size,
                instr_vocab_size,
                max_recipe_len=max_recipe_len,
                ingr_eos_value=ingr_eos_value,
            )
        else:
            raise NotImplementedError(f"Task {task} is not implemented yet")

        self.task = task
        self.optimization = optim_config

        if self.task != TaskType.ingr2recipe:
            # check if pretrained models
            self.pretrained_imenc = image_encoder_config.pretrained
            self.pretrained_ingrpred = (
                ingr_pred_config.load_pretrained_from != "None"
            )  ## TODO: load model when pretrained

        # metrics to track at validation time
        if self.task != TaskType.ingr2recipe:
            self.o_f1 = DistributedF1(
                which_f1="o_f1",
                pad_value=self.model.ingr_vocab_size - 1,
                remove_eos=self.model.ingr_predictor.remove_eos,
                dist_sync_on_step=True,
            )
            self.c_f1 = DistributedF1(
                which_f1="c_f1",
                pad_value=self.model.ingr_vocab_size - 1,
                remove_eos=self.model.ingr_predictor.remove_eos,
                dist_sync_on_step=True,
            )
            self.i_f1 = DistributedF1(
                which_f1="i_f1",
                pad_value=self.model.ingr_vocab_size - 1,
                remove_eos=self.model.ingr_predictor.remove_eos,
                dist_sync_on_step=True,
            )
            self.val_losses = DistributedValLosses(
                weights=self.optimization.loss_weights,
                monitor_ingr_losses=ingr_pred_config.freeze,
                dist_sync_on_step=True,
            )
        else:
            self.val_losses = DistributedValLosses(
                weights=self.optimization.loss_weights, dist_sync_on_step=True
            )
        if self.task != TaskType.im2ingr:
            self.perplexity = DistributedMetric()

    def forward(
        self,
        split: str,
        image: Optional[torch.Tensor] = None,
        ingredients: Optional[torch.Tensor] = None,
        recipe: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ):
        if self.task == TaskType.im2ingr:
            out = self.model(
                image=image,
                target_ingredients=ingredients,
                compute_losses=compute_losses,
                compute_predictions=compute_predictions,
            )
        elif self.task == TaskType.im2recipe:
            out = self.model(
                image=image,
                target_recipe=recipe,
                target_ingredients=ingredients,
                use_ingr_pred=True if split == "test" else False,
                compute_losses=compute_losses,
                compute_predictions=compute_predictions,
            )
        elif self.task == TaskType.ingr2recipe:
            out = self.model(
                ingredients=ingredients,
                target_recipe=recipe,
                compute_losses=compute_losses,
                compute_predictions=compute_predictions,
            )
        return out[0], out[1:]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(compute_losses=True, split="train", **batch)
        return out[0]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self._evaluation_step(batch, prefix="val")
        return out

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self._evaluation_step(batch, prefix="test")
        return out

    def _evaluation_step(self, batch: Dict[str, torch.Tensor], prefix: str):
        if self.task == TaskType.im2ingr:
            out = self(
                **batch, split=prefix, compute_predictions=True, compute_losses=True
            )
            out[0]["n_samples"] = batch["image"].shape[0]
        elif self.task in [TaskType.im2recipe, TaskType.ingr2recipe]:
            out = self(
                **batch, split=prefix, compute_predictions=False, compute_losses=True,
            )
            out[0]["n_samples"] = batch["recipe"].shape[0]

        if self.task in [TaskType.im2recipe, TaskType.im2ingr]:
            out[0]["ingr_pred"] = out[1][0]
            out[0]["ingr_gt"] = batch["ingredients"]
        return out[0]

    def training_step_end(self, losses: Dict[str, torch.Tensor]):
        """
        Average the loss across all GPUs and combine these losses together as an overall loss
        """

        # avg losses across gpus
        for k in losses.keys():
            losses[k] = losses[k].mean()

        total_loss = 0
        loss_weights = self.optimization.loss_weights

        if "label_loss" in losses.keys():
            self._log_loss("label_loss", losses["label_loss"])
            total_loss += losses["label_loss"] * loss_weights["label_loss"]

        if "cardinality_loss" in losses.keys():
            self._log_loss("cardinality_loss", losses["cardinality_loss"])
            total_loss += losses["cardinality_loss"] * loss_weights["cardinality_loss"]

        if "eos_loss" in losses.keys():
            self._log_loss("eos_loss", losses["eos_loss"])
            total_loss += losses["eos_loss"] * loss_weights["eos_loss"]

        if "recipe_loss" in losses.keys():
            self._log_loss("recipe_loss", losses["recipe_loss"])
            total_loss += losses["recipe_loss"] * loss_weights["recipe_loss"]

        self._log_loss("train_loss", total_loss)
        return total_loss

    def _log_loss(self, key: str, value: Any):
        self.log(
            key, value, on_step=True, on_epoch=True, prog_bar=True, logger=True,
        )

    def validation_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def test_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def _evaluation_step_end(self, step_output: Dict[str, Any]):
        """
        Update distributed metrics
        """
        # compute ingredient metrics
        if self.task in [TaskType.im2ingr, TaskType.im2recipe]:
            # update f1 metrics
            if step_output["ingr_pred"] is not None:
                self.o_f1(step_output["ingr_pred"], step_output["ingr_gt"])
                self.c_f1(step_output["ingr_pred"], step_output["ingr_gt"])
                self.i_f1(step_output["ingr_pred"], step_output["ingr_gt"])

        # update recipe metrics
        if self.task in [TaskType.im2recipe, TaskType.ingr2recipe]:
            self.perplexity(torch.exp(step_output["recipe_loss"]))

        # update losses
        self.val_losses(step_output)

    def validation_epoch_end(self, out):
        self._eval_epoch_end(split="val")

    def test_epoch_end(self, out):
        self._eval_epoch_end(split="test")

    def _eval_epoch_end(self, split: str):
        """
        Compute and log metrics/losses
        """
        if self.task == TaskType.im2ingr or (
            self.task == TaskType.im2recipe and split == "test"
        ):
            self.log(f"{split}_o_f1", self.o_f1.compute())
            self.log(f"{split}_c_f1", self.c_f1.compute())
            self.log(f"{split}_i_f1", self.i_f1.compute())

        if self.task in [TaskType.im2recipe, TaskType.ingr2recipe]:
            self.log(f"{split}_perplexity", self.perplexity.compute())

        val_losses = self.val_losses.compute()
        for k, v in val_losses.items():
            self.log(f"{split}_{k}", v)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.create_parameter_groups(),
            lr=self.optimization.lr,
            weight_decay=self.optimization.weight_decay,
        )
        scheduler = {
            "scheduler": ExponentialLR(optimizer, self.optimization.lr_decay_rate),
            "interval": "epoch",
            "frequency": self.optimization.lr_decay_every,
        }
        return [optimizer], [scheduler]

    def create_parameter_groups(self):
        opt_arguments = []
        pretrained_lr = self.optimization.lr * self.optimization.scale_lr_pretrained

        if hasattr(self.model, "image_encoder"):
            params_imenc = filter(
                lambda p: p.requires_grad, self.model.image_encoder.parameters()
            )
            num_params_imenc = sum([p.numel() for p in params_imenc])
            print(
                f"Number of trainable parameters in the image encoder is {num_params_imenc}."
            )

            if num_params_imenc > 0:
                opt_arguments += [
                    {
                        "params": params_imenc,
                        "lr": pretrained_lr
                        if self.pretrained_imenc
                        else self.optimization.lr,
                    }
                ]

        if hasattr(self.model, "ingr_encoder"):
            params_ingr_enc = filter(
                lambda p: p.requires_grad, self.model.ingr_encoder.parameters()
            )
            num_params_ingr_enc = sum([p.numel() for p in params_ingr_enc])
            print(
                f"Number of trainable parameters in the ingredient encoder is {num_params_ingr_enc}."
            )

            if num_params_ingr_enc > 0:
                opt_arguments += [
                    {"params": params_ingr_enc, "lr": self.optimization.lr,}
                ]

        if hasattr(self.model, "ingr_predictor"):
            params_ingrpred = filter(
                lambda p: p.requires_grad, self.model.ingr_predictor.parameters()
            )
            num_params_ingrpred = sum([p.numel() for p in params_ingrpred])
            print(
                f"Number of trainable parameters in the ingredient predictor is {num_params_ingrpred}."
            )

            if num_params_ingrpred > 0:
                opt_arguments += [
                    {
                        "params": params_ingrpred,
                        "lr": pretrained_lr
                        if self.pretrained_ingrpred
                        else self.optimization.lr,
                    }
                ]

        if hasattr(self.model, "recipe_gen"):
            params_recgen = filter(
                lambda p: p.requires_grad, self.model.recipe_gen.parameters()
            )
            num_params_recgen = sum([p.numel() for p in params_recgen])
            print(
                f"Number of trainable parameters in the recipe generator is {num_params_recgen}."
            )

            if num_params_recgen > 0:
                opt_arguments += [{"params": params_recgen, "lr": self.optimization.lr}]

        return opt_arguments
