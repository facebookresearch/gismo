# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple, List

import numpy as np
import torch
import torch.distributed as dist
from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.utils.visualisation.recipe_utils import recipe_length, recipe_lexical_diversity
import pytorch_lightning as pl


class DistributedAverageWithMemory(pl.metrics.Metric):
    """
    Metric to compute the average of a value, distributed among several workers
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # TODO - find a way to synchronize these values to truly collect
        #  the global extremum and not local extremum
        self.memory = []
        self.recipe_ids = []
        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, quantity: torch.Tensor, recipe_ids: List[str]):
        self.memory.extend(quantity.cpu().numpy())
        self.recipe_ids.extend(recipe_ids)
        self.quantity += quantity.sum()
        self.n_samples += quantity.numel()

    def compute(self, name: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        avg = self.quantity / self.n_samples
        std = self._compute_distributed_std(avg)
        self._print_extreme_scores(name)
        return avg, std

    def _compute_distributed_std(self, avg: torch.Tensor) -> torch.Tensor:
        sum_of_squares = (torch.tensor(self.memory, device=avg.device) - avg).pow_(2).sum()
        dist.all_reduce(sum_of_squares)
        std = (sum_of_squares / self.n_samples).sqrt()
        return std

    def _print_extreme_scores(self, name: str):
        memory = np.array(self.memory)
        recipe_ids = np.array(self.recipe_ids)
        indices = np.argsort(memory)
        low_indices = indices[:20]
        high_indices = indices[-20:]
        if dist.get_rank() == 0:
            print(f"Lowest indices ({name}):", list(recipe_ids[low_indices]))
            print(f"Lowest values ({name}):", list(memory[low_indices]))
            print(f"Highest indices ({name}):", list(recipe_ids[high_indices]))
            print(f"Highest values ({name}):", list(memory[high_indices]))


class RecipeFeaturesMetric(DistributedAverageWithMemory):
    def __init__(self, instr_vocab: Vocabulary, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.instr_vocab = instr_vocab

    def add(self, recipes: torch.Tensor, recipe_ids: List[str]) -> torch.Tensor:
        features = self._compute_features(recipes)
        self.update(features, recipe_ids)
        return features


class RecipeLengthMetric(RecipeFeaturesMetric):
    def _compute_features(self, recipes: torch.Tensor) -> torch.Tensor:
        lengths = [
            recipe_length(recipes[i], self.instr_vocab)
            for i in range(recipes.shape[0])
        ]
        return torch.tensor(lengths, dtype=torch.int64, device=recipes.device)


class RecipeVocabDiversity(RecipeFeaturesMetric):
    def _compute_features(self, recipes: torch.Tensor) -> torch.Tensor:
        diversities = [
            recipe_lexical_diversity(recipes[i], self.instr_vocab)
            for i in range(recipes.shape[0])
        ]
        return torch.tensor(diversities, dtype=torch.int64, device=recipes.device)
