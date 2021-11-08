# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.utils.metrics import DistributedAverage
from inv_cooking.utils.visualisation.recipe_utils import recipe_length, recipe_lexical_diversity


class RecipeFeaturesMetric(DistributedAverage):
    def __init__(self, instr_vocab: Vocabulary, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.instr_vocab = instr_vocab

    def add(self, recipes: torch.Tensor) -> torch.Tensor:
        features = self._compute_features(recipes)
        self.update(features)
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
