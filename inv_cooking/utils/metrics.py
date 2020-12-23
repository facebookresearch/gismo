# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytorch_lightning as pl
import torch

from inv_cooking.models.ingredients_predictor import label2_k_hots


class DistributedF1(pl.metrics.Metric):
    def __init__(self, which_f1: str, pad_value: int, remove_eos: bool, dist_sync_on_step: bool=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        assert which_f1 in ["i_f1", "o_f1", "c_f1"]

        self.which_f1 = which_f1
        self.pad_value = pad_value
        self.remove_eos = remove_eos

        if self.which_f1 == "i_f1":
            self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        elif self.which_f1 == "o_f1":
            self.add_state("tp", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("pp", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("ap", default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("tp", default=torch.zeros(self.pad_value-1), dist_reduce_fx="sum")
            self.add_state("pp", default=torch.zeros(self.pad_value-1), dist_reduce_fx="sum")
            self.add_state("ap", default=torch.zeros(self.pad_value-1), dist_reduce_fx="sum")            
        
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        assert pred.shape == gt.shape

        # convert model predictions and targets to k-hots
        y_pred = label2_k_hots(
            pred,
            self.pad_value,
            self.remove_eos,
        )
        y_true = label2_k_hots(
            gt,
            self.pad_value,
            self.remove_eos,
        )

        if self.which_f1 == "i_f1":
            self.tp = (y_pred * y_true).sum(1)
            self.pp = y_pred.sum(1)
            self.ap = y_true.sum(1)
            self.f1 += self._computef1().sum()
            self.n_samples += y_pred.shape[0]
        elif self.which_f1 == "o_f1":
            self.tp += (y_pred * y_true).sum()
            self.pp += y_pred.sum()
            self.ap += y_true.sum()
        elif self.which_f1 == "c_f1":
            self.tp += (y_pred * y_true).sum(0)
            self.pp += y_pred.sum(0)
            self.ap += y_true.sum(0)

    def _computef1(self, eps: float = 1e-8):
        pre = (self.tp + eps) / (self.pp + eps)
        rec = (self.tp + eps) / (self.ap + eps)
        f1 = 2 * (pre * rec) / (pre + rec)

        return f1

    def compute(self):
        if self.which_f1 == "c_f1":
            f1 = self._computef1()
            f1 = f1.mean()
        elif self.which_f1 == "o_f1":
            f1 = self._computef1()
        elif self.which_f1 == "i_f1":
            f1 = self.f1 / self.n_samples
            
        return f1


class DistributedMetric(pl.metrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
    def update(self, quantity: torch.Tensor):
            self.quantity += quantity.sum()
            self.n_samples += quantity.shape[0]

    def compute(self):
        return self.quantity / self.n_samples


class DistributedValLosses(pl.metrics.Metric):
    def __init__(self, weights, monitor_ingr_losses=False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.weights = weights

        if monitor_ingr_losses and "label_loss" in self.weights.keys() and self.weights["label_loss"] > 0:
            self.add_state("label_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if monitor_ingr_losses and "cardinality_loss" in self.weights.keys() and self.weights["cardinality_loss"] > 0:
            self.add_state("cardinality_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if monitor_ingr_losses and "eos_loss" in self.weights.keys() and self.weights["eos_loss"] > 0:
            self.add_state("eos_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        if "recipe_loss" in self.weights.keys() and self.weights["recipe_loss"] > 0:
            self.add_state("recipe_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")     

        self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")                    
        
    def update(self, quantities):
        if hasattr(self, "label_loss"):
            self.label_loss += quantities["label_loss"].sum()
        if hasattr(self, "cardinality_loss"):
            self.cardinality_loss += quantities["cardinality_loss"].sum()
        if hasattr(self, "eos_loss"):
            self.eos_loss += quantities["eos_loss"].sum()
        if hasattr(self, "recipe_loss"):
            self.recipe_loss += quantities["recipe_loss"].sum()

        self.n_samples += torch.Tensor(quantities["n_samples"]).sum()

    def compute(self):
        ret_dict = {}
        ret_dict["total_loss"] = 0
        if hasattr(self, "label_loss"):
            ret_dict["label_loss"] = self.label_loss / self.n_samples
            ret_dict["total_loss"] += self.weights["label_loss"] * ret_dict["label_loss"]
        if hasattr(self, "cardinality_loss"):
            ret_dict["cardinality_loss"] = self.cardinality_loss / self.n_samples
            ret_dict["total_loss"] += self.weights["cardinality_loss"] * ret_dict["cardinality_loss"]
        if hasattr(self, "eos_loss"):
            ret_dict["eos_loss"] = self.eos_loss / self.n_samples
            ret_dict["total_loss"] += self.weights["eos_loss"] * ret_dict["eos_loss"]
        if hasattr(self, "recipe_loss"):
            ret_dict["recipe_loss"] = self.recipe_loss / self.n_samples
            ret_dict["total_loss"] += self.weights["recipe_loss"] * ret_dict["recipe_loss"]

        return ret_dict