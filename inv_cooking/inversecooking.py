from typing import Optional, Any, List, Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ExponentialLR

from .config import TaskType, RecipeGeneratorConfig, OptimizationConfig, ImageEncoderConfig
from .models.im2ingr import Im2Ingr
from .models.im2recipe import Im2Recipe
from .models.ingr2recipe import Ingr2Recipe
from .models.ingredients_predictor import label2_k_hots
from .utils.metrics import OverallErrorCounts


class LitInverseCooking(pl.LightningModule):
    def __init__(
        self,
        task: TaskType,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: DictConfig,
        recipe_gen_config: RecipeGeneratorConfig,
        optim_config: OptimizationConfig,
        dataset_name: str,  ## TODO: check if needed at all
        maxnumlabels: int,
        maxrecipelen: int,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        ingr_eos_value,
    ):
        super().__init__()

        if task == TaskType.im2ingr:
            self.model = Im2Ingr(
                image_encoder_config,
                ingr_pred_config,
                ingr_vocab_size,
                dataset_name,
                maxnumlabels,
                ingr_eos_value,
            )
        elif task == TaskType.im2recipe:
            self.model = Im2Recipe(
                image_encoder_config,
                ingr_pred_config,
                recipe_gen_config,
                ingr_vocab_size,
                instr_vocab_size,
                dataset_name,
                maxnumlabels,
                maxrecipelen,
                ingr_eos_value,
            )
        elif task == TaskType.ingr2recipe:
            self.model = Ingr2Recipe(
                recipe_gen_config,
                ingr_vocab_size,
                instr_vocab_size,
                max_recipe_len=maxrecipelen,
                ingr_eos_value=ingr_eos_value,
            )
        else:
            raise NotImplementedError(f"Task {task} is not implemented yet")

        self.task = task
        if self.task != TaskType.ingr2recipe:
            self.pretrained_imenc = image_encoder_config.pretrained
            self.pretrained_ingrpred = (
                ingr_pred_config.load_pretrained_from != "None"
            )  ## TODO: load model when pretrained
        self.lr = optim_config.lr
        self.scale_lr_pretrained = optim_config.scale_lr_pretrained
        self.lr_decay_rate = optim_config.lr_decay_rate
        self.lr_decay_every = optim_config.lr_decay_every
        self.weight_decay = optim_config.weight_decay
        self.loss_weights = optim_config.loss_weights

        self.overall_error_counts = OverallErrorCounts()
        self.overall_error_counts.reset(overall=True)

    def forward(
        self,
        split: str,
        img: Optional[torch.Tensor] = None,
        ingr_gt: Optional[torch.Tensor] = None,
        recipe_gt: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ):
        if self.task == TaskType.im2ingr:
            out = self.model(
                image=img,
                label_target=ingr_gt,
                compute_losses=compute_losses,
                compute_predictions=compute_predictions,
            )
        elif self.task == TaskType.im2recipe:
            out = self.model(
                image=img,
                recipe_gt=recipe_gt,
                ingr_gt=ingr_gt,
                use_ingr_pred=True if split == "test" else False,
                compute_losses=compute_losses,
                compute_predictions=compute_predictions,
            )
        elif self.task == TaskType.ingr2recipe:
             out = self.model(
                recipe_gt=recipe_gt,
                ingr_gt=ingr_gt,
                compute_losses=compute_losses,
                compute_predictions=compute_predictions,
            )           

        return out[0], out[1:]

    def training_step(self, batch, batch_idx: int):
        out = self(compute_losses=True, split="train", **batch)
        return out[0]

    def validation_step(self, batch, batch_idx: int):
        return self._evaluation_step(batch, prefix="val")

    def test_step(self, batch, batch_idx: int):
        return self._evaluation_step(batch, prefix="test")

    def _evaluation_step(self, batch, prefix: str):
        metrics = {}

        # get model outputs
        if self.task == TaskType.im2ingr:
            out = self(img=batch["img"], split=prefix, compute_predictions=True)
        elif self.task in [TaskType.im2recipe, TaskType.ingr2recipe]:
            if prefix == "val":
                out = self(
                    **batch,
                    split=prefix,
                    compute_predictions=False,
                    compute_losses=True,
                )
            elif prefix == "test":
                out = self(
                    **batch,
                    split=prefix,
                    compute_predictions=False,
                    compute_losses=True,
                )

        # compute ingredient metrics
        if self.task == TaskType.im2ingr or (TaskType.im2recipe and out[1][0] is not None):
            # convert model predictions and targets to k-hots
            pred_k_hots = label2_k_hots(
                out[1][0],
                self.model.ingr_vocab_size - 1,
                remove_eos=not self.model.ingr_predictor.is_decoder_ff,
            )
            target_k_hots = label2_k_hots(
                batch["ingr_gt"],
                self.model.ingr_vocab_size - 1,
                remove_eos=not self.model.ingr_predictor.is_decoder_ff,
            )

            # update overall and per class error counts
            self.overall_error_counts.update(
                pred_k_hots,
                target_k_hots,
                which_metrics=["o_f1", "c_f1", "i_f1"],
            )

            # compute i_f1 metric
            metrics = self.overall_error_counts.compute_metrics(which_metrics=["i_f1"])

        # compute recipe metrics
        if self.task in [TaskType.im2recipe, TaskType.ingr2recipe]:
            metrics["perplexity"] = torch.exp(out[0]["recipe_loss"])

        # save n_samples
        if self.task == TaskType.im2ingr:
            metrics["n_samples"] = batch["img"].shape[0]
        else:
            metrics["n_samples"] = batch["ingr_gt"].shape[0]

        return metrics

    def validation_epoch_end(self, valid_step_outputs: List[Any]):
        self._eval_epoch_end(valid_step_outputs, split="val")

    def test_epoch_end(self, test_step_outputs: List[Any]):
        self._eval_epoch_end(test_step_outputs, split="test")

    def _eval_epoch_end(self, eval_step_outputs: List[Any], split: str):
        s = sum(self.overall_error_counts.counts.values())
        if (isinstance(s, int) and s > 0) or (not isinstance(s, int) and s.any()):
            # compute validation set metrics
            overall_metrics = self.overall_error_counts.compute_metrics(["o_f1", "c_f1"])
            self.log(f"{split}_o_f1", overall_metrics["o_f1"])
            self.log(f"{split}_c_f1", overall_metrics["c_f1"])
            self.overall_error_counts.reset(overall=True)

        # init avg metrics to 0
        avg_metrics = dict(
            zip(eval_step_outputs[0].keys(), [0] * len(eval_step_outputs[0].keys()))
        )

        # update avg metrics
        for out in eval_step_outputs:
            for k in out.keys():
                avg_metrics[k] += out[k]

        # log all validation metrics
        for k in avg_metrics.keys():
            if k != "n_samples":
                self.log(f"{split}_{k}", avg_metrics[k] / avg_metrics["n_samples"])

    def training_step_end(self, losses: Dict[str, torch.Tensor]):
        """
        Average the loss across all GPUs and combine these losses together as an overall loss
        """

        total_loss = 0

        # avg losses across gpus
        for k in losses.keys():
            losses[k] = losses[k].mean()

        if "label_loss" in losses.keys():
            self.log(
                "label_loss",
                losses["label_loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            total_loss += losses["label_loss"] * self.loss_weights["label_loss"]

        if "cardinality_loss" in losses.keys():
            self.log(
                "cardinality_loss",
                losses["cardinality_loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            total_loss += (
                losses["cardinality_loss"] * self.loss_weights["cardinality_loss"]
            )

        if "eos_loss" in losses.keys():
            self.log(
                "eos_loss",
                losses["eos_loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            total_loss += losses["eos_loss"] * self.loss_weights["eos_loss"]

        if "recipe_loss" in losses.keys():
            self.log(
                "recipe_loss",
                losses["recipe_loss"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            total_loss += losses["recipe_loss"] * self.loss_weights["recipe_loss"]

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return total_loss

    def validation_step_end(self, metrics):
        valid_step_outputs = self._evaluation_step_end(metrics)
        return valid_step_outputs

    def test_step_end(self, metrics):
        test_step_outputs = self._evaluation_step_end(metrics)
        return test_step_outputs

    def _evaluation_step_end(self, metrics: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Sum metrics withing mini-batches and reset the per sample error counts
        """
        eval_step_outputs = {}
        for k in metrics.keys():
            eval_step_outputs[k] = sum(metrics[k])
        self.overall_error_counts.reset(per_image=True)
        return eval_step_outputs

    def configure_optimizers(self):
        opt_arguments = []
        pretrained_lr = self.lr * self.scale_lr_pretrained

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
                        "lr": pretrained_lr if self.pretrained_imenc else self.lr,
                    }
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
                        "lr": pretrained_lr if self.pretrained_ingrpred else self.lr,
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
                opt_arguments += [{"params": params_recgen, "lr": self.lr}]


        optimizer = torch.optim.Adam(
            opt_arguments, lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = {
            "scheduler": ExponentialLR(optimizer, self.lr_decay_rate),
            "interval": "epoch",
            "frequency": self.lr_decay_every,
        }

        return [optimizer], [scheduler]
