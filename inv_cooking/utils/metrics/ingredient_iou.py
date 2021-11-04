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

    def __init__(self, ingr_vocab: Vocabulary, instr_vocab: Vocabulary):
        super().__init__()
        self.ingr_vocab = ingr_vocab
        self.instr_vocab = instr_vocab

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
