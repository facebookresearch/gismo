from typing import List, Set

import torch

from inv_cooking.datasets.vocabulary import Vocabulary


def recipe_to_text(prediction: torch.Tensor, vocab: Vocabulary):
    sentence = ""
    for i in prediction.cpu().numpy():
        word = vocab.idx2word.get(i)
        if word == "<end>":
            return sentence

        if word == "<eoi>":
            sentence += "\n"
        elif word != "<start>":
            sentence += " " + word
    return sentence


def format_recipe(generated_recipe: str):
    generated_recipe = generated_recipe.strip()
    generated_recipe = generated_recipe.replace(" .", ".")
    generated_recipe = generated_recipe.replace(" ,", ",")
    generated_recipe = generated_recipe.replace("\n", "\n - ")
    return generated_recipe


def ingredients_to_text(prediction: torch.Tensor, vocab: Vocabulary,) -> List[str]:
    ingredient_list = []
    for i in prediction.cpu().numpy():
        word = vocab.idx2word.get(i)
        if word != "<pad>":
            if isinstance(word, list):
                ingredient_list.append(word[0])
            else:
                ingredient_list.append(word)
    return ingredient_list


def split_ingredient_components(ingredients: List[str]) -> List[Set[str]]:
    components = []
    for ingredient in ingredients:
        components.append(set(ingredient.split("_")))
    return components
