from typing import List, Sequence

import numpy as np
import torch

from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.utils.metrics import DistributedAverage
from inv_cooking.utils.visualisation.recipe_utils import (
    ingredients_to_text,
    recipe_to_text,
    split_ingredient_components,
)


class IngredientIntersection(DistributedAverage):
    """
    Metric to compute how many ingredients appear in the recipe
    """

    def __init__(self, ingr_vocab: Vocabulary, instr_vocab: Vocabulary, threshold: float = 1.0):
        super().__init__()
        self.ingr_vocab = ingr_vocab
        self.instr_vocab = instr_vocab
        self.threshold = threshold
        self._compute_graph()

    def _compute_graph(self):
        self.ingr_id_to_word = {}
        self.word_to_ingr_id = {}
        for ingr_id, full_ingredients in self.ingr_vocab.idx2word.items():
            for choice_id, full_ingredient in enumerate(full_ingredients):
                for ingr_component in full_ingredient.split("_"):
                    # TODO - filter the 1% and things like that?
                    key = ingr_id, choice_id
                    self.ingr_id_to_word.setdefault(key, set()).add(ingr_component)
                    self.word_to_ingr_id.setdefault(ingr_component, []).append(key)

    def compute_iou(self, ingredients: torch.Tensor, words: Sequence[str]):

        # print(words)

        # Go through the text and count the number of components associated to a target
        targets = set(ingredients.cpu().numpy())
        target_contributions = {}
        for word in words:
            for key in self.word_to_ingr_id.get(word, []):
                ingr_id, choice_id = key
                if ingr_id in targets:
                    target_contributions.setdefault(key, set()).add(word)

        # TODO - use a priority queue to remove matches from the top? a best matching
        #  strategy for the ingredients, with no preference positive or negative

        # Go through each target, count if enough component match, and if so
        # consider the target as identified + remove all words that allowed
        # to identify that target
        identified_positive = set()
        visited_words = set()
        for key, words in target_contributions.items():
            words = words - visited_words
            if len(words) >= self.threshold * len(self.ingr_id_to_word[key]):
                identified_positive.add(key[0])
                for word in words:
                    # Avoid matching several ingredients with the same word
                    visited_words.add(word)

        # Now go through the words again, ignore the one that helped identify
        # a positive ingredient, and see how many negative examples are matched
        found_negatives = {}
        for word in words:
            if word not in visited_words:
                for key in self.word_to_ingr_id.get(word, []):
                    ingr_id, choice_id = key
                    if ingr_id not in targets:
                        found_negatives.setdefault(key, set()).add(word)

        identified_negative = set()
        for key, words in found_negatives.items():
            words = words - visited_words
            if len(words) >= self.threshold * len(self.ingr_id_to_word[key]):
                identified_negative.add(key[0])
                for word in words:
                    # Avoid matching several ingredients with the same word
                    visited_words.add(word)

        # Compute the Intersection Over Union
        # print(identified_positive)
        # print(identified_negative)
        intersection = len(identified_positive)
        union = len(identified_positive) + len(identified_negative)
        if union > 0:
            return intersection / union
        else:
            return 0.0

    def add(self, ingredients: torch.tensor, recipes: torch.Tensor):
        results = []
        batch_size = recipes.shape[0]
        for i in range(batch_size):
            raw_recipe_text = recipe_to_text(recipes[i], self.instr_vocab)
            raw_recipe_words = list({
                w for line in raw_recipe_text.splitlines() for w in line.split(" ")
            })
            iou = self.compute_iou(ingredients[i], raw_recipe_words)
            results.append(iou)
        results = torch.tensor(results, device=ingredients.device)
        self.update(results)
        return results

    """
    def add(self, ingredients: torch.tensor, recipes: torch.Tensor):
        results = []
        batch_size = recipes.shape[0]
        for i in range(batch_size):
            ingredient_list = ingredients_to_text(ingredients[i], self.ingr_vocab)
            raw_recipe_text = recipe_to_text(recipes[i], self.instr_vocab)
            raw_recipe_words = {
                w for line in raw_recipe_text.splitlines() for w in line.split(" ")
            }
            ingredients_components = split_ingredient_components(ingredient_list)
            mean_matches = np.array(
                [
                    1.0
                    if len(components & raw_recipe_words) / len(components) >= 0.5
                    else 0.0
                    for components in ingredients_components
                ]
            ).mean()
            results.append(mean_matches)
        results = torch.tensor(results, device=ingredients.device)
        self.update(results)
        return results
    """
