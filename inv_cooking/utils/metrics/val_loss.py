# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from inv_cooking.utils.metrics.average import DistributedCompositeAverage


class DistributedValLosses(DistributedCompositeAverage):
    """
    Metric to compute the average of the validation losses tracked for recipe
    """

    def __init__(
        self,
        weights: Dict[str, float],
        monitor_ingr_losses=False,
        dist_sync_on_step=False,
    ):
        super().__init__(
            weights=self.filter_weights(weights, monitor_ingr_losses),
            total="total_loss",
            dist_sync_on_step=dist_sync_on_step,
        )

    @staticmethod
    def filter_weights(weights: Dict[str, float], monitor_ingr_losses: bool):
        filtered_weights = {}
        if monitor_ingr_losses and "label_loss" in weights:
            filtered_weights["label_loss"] = weights["label_loss"]
        if monitor_ingr_losses and "cardinality_loss" in weights:
            filtered_weights["cardinality_loss"] = weights["cardinality_loss"]
        if monitor_ingr_losses and "eos_loss" in weights:
            filtered_weights["eos_loss"] = weights["eos_loss"]
        if "recipe_loss" in weights:
            filtered_weights["recipe_loss"] = weights["recipe_loss"]
        return filtered_weights
