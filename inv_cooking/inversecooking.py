import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import ExponentialLR

from .config import TaskType
from .models.im2ingr import Im2Ingr
from .models.im2recipe import Im2Recipe
from .models.ingredients_predictor import label2_k_hots
from .utils.metrics import compute_metrics, update_error_counts


class LitInverseCooking(pl.LightningModule):
    def __init__(
        self,
        task: TaskType,
        im_args,
        ingrpred_args,
        recipegen_args,
        optim_args,
        dataset_name,  ## TODO: check if needed at all
        maxnumlabels,
        maxrecipelen,
        ingr_vocab_size,
        instr_vocab_size,
        ingr_eos_value,
    ):
        super().__init__()

        if task == TaskType.im2ingr:
            self.model = Im2Ingr(
                im_args,
                ingrpred_args,
                ingr_vocab_size,
                dataset_name,
                maxnumlabels,
                ingr_eos_value,
            )
        elif task == TaskType.im2recipe:
            self.model = Im2Recipe(
                im_args,
                ingrpred_args,
                recipegen_args,
                ingr_vocab_size,
                instr_vocab_size,
                dataset_name,
                maxnumlabels,
                maxrecipelen,
                ingr_eos_value,
            )
        else:
            raise NotImplementedError(f"Task {task} is not implemented yet")

        self.task = task
        self.pretrained_imenc = im_args.pretrained
        self.pretrained_ingrpred = (
            ingrpred_args.load_pretrained_from != "None"
        )  ## TODO: load model when pretrained
        self.lr = optim_args.lr
        self.scale_lr_pretrained = optim_args.scale_lr_pretrained
        self.lr_decay_rate = optim_args.lr_decay_rate
        self.lr_decay_every = optim_args.lr_decay_every
        self.weight_decay = optim_args.weight_decay
        self.loss_weights = optim_args.loss_weights

        self._reset_error_counts(overall=True)

    def forward(
        self,
        img,
        split,
        ingr_gt=None,
        recipe_gt=None,
        compute_losses=False,
        compute_predictions=False,
    ):
        if self.task == TaskType.im2ingr:
            out = self.model(
                img=img,
                label_target=ingr_gt,
                compute_losses=compute_losses,
                compute_predictions=compute_predictions,
            )
        elif self.task == TaskType.im2recipe:
            out = self.model(
                img=img,
                recipe_gt=recipe_gt,
                ingr_gt=ingr_gt,
                use_ingr_pred=True if split == "test" else False,
                compute_losses=compute_losses,
                compute_predictions=compute_predictions,
            )

        return out[0], out[1:]

    def training_step(self, batch, batch_idx):
        out = self(compute_losses=True, split="train", **batch)
        return out[0]

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_eval(batch, batch_idx, "val")
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self._shared_eval(batch, batch_idx, "test")
        return metrics

    def _shared_eval(self, batch, batch_idx, prefix):

        metrics = {}

        # get model outputs
        if self.task == TaskType.im2ingr:
            out = self(batch["img"], split=prefix, compute_predictions=True)
        elif self.task == TaskType.im2recipe:
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
        if out[1][0] is not None:
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
            update_error_counts(
                self.overall_error_counts,
                pred_k_hots,
                target_k_hots,
                which_metrics=["o_f1", "c_f1", "i_f1"],
            )

            # compute i_f1 metric
            metrics = compute_metrics(self.overall_error_counts, which_metrics=["i_f1"])

        # compute recipe metrics
        if self.task == TaskType.im2recipe:
            metrics["perplexity"] = torch.exp(out[0]["recipe_loss"])

        # save n_samples
        metrics["n_samples"] = batch["img"].shape[0]

        return metrics

    def validation_epoch_end(self, valid_step_outputs):
        self.eval_epoch_end(valid_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        self.eval_epoch_end(test_step_outputs, "test")

    def eval_epoch_end(self, eval_step_outputs, split):
        s = sum(self.overall_error_counts.values())
        if (isinstance(s, int) and s > 0) or (not isinstance(s, int) and s.any()):
            # compute validation set metrics
            overall_metrics = compute_metrics(
                self.overall_error_counts, ["o_f1", "c_f1"]
            )
            self.log(f"{split}_o_f1", overall_metrics["o_f1"])
            self.log(f"{split}_c_f1", overall_metrics["c_f1"])
            self._reset_error_counts(overall=True)

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

    def training_step_end(self, losses):

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
        valid_step_outputs = self._shared_eval_step_end(metrics)
        return valid_step_outputs

    def test_step_end(self, metrics):
        test_step_outputs = self._shared_eval_step_end(metrics)
        return test_step_outputs

    def _shared_eval_step_end(self, metrics):
        eval_step_outputs = {}

        # sum metric within mini-batches
        for k in metrics.keys():
            eval_step_outputs[k] = sum(metrics[k])

        # reset per sample error counts
        self._reset_error_counts(perimage=True)

        return eval_step_outputs

    def _reset_error_counts(self, overall=False, perimage=False):
        # reset all error counts (done at the end of each epoch)
        if overall:
            self.overall_error_counts = {
                "c_tp": 0,
                "c_fp": 0,
                "c_fn": 0,
                "c_tn": 0,
                "o_tp": 0,
                "o_fp": 0,
                "o_fn": 0,
                "i_tp": 0,
                "i_fp": 0,
                "i_fn": 0,
            }
        # reset per sample error counts (done at the end of each iteration)
        if perimage:
            self.overall_error_counts["i_tp"] = 0
            self.overall_error_counts["i_fp"] = 0
            self.overall_error_counts["i_fn"] = 0

    def configure_optimizers(self):
        params_imenc = filter(
            lambda p: p.requires_grad, self.model.image_encoder.parameters()
        )
        num_params_imenc = sum([p.numel() for p in params_imenc])
        params_ingrpred = filter(
            lambda p: p.requires_grad, self.model.ingr_predictor.parameters()
        )
        num_params_ingrpred = sum([p.numel() for p in params_ingrpred])

        print(
            f"Number of trainable parameters in the image encoder is {num_params_imenc}."
        )
        print(
            f"Number of trainable parameters in the ingredient predictor is {num_params_ingrpred}."
        )

        pretrained_lr = self.lr * self.scale_lr_pretrained

        opt_arguments = []

        if num_params_imenc > 0:
            opt_arguments += [
                {
                    "params": params_imenc,
                    "lr": pretrained_lr if self.pretrained_imenc else self.lr,
                }
            ]

        if num_params_ingrpred > 0:
            opt_arguments += [
                {
                    "params": params_ingrpred,
                    "lr": pretrained_lr if self.pretrained_ingrpred else self.lr,
                }
            ]

        if self.task == TaskType.im2recipe:
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
