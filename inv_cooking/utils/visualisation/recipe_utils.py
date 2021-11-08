# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
        elif word not in {"<start>", "<pad>"}:
            sentence += " " + word
    return sentence


def recipe_length(prediction: torch.Tensor, vocab: Vocabulary) -> int:
    length = 0
    for i in prediction.cpu().numpy():
        word = vocab.idx2word.get(i)
        if word == "<end>":
            return length

        if word not in {"<start>", "<pad>"}:
            length += 1
    return length


def recipe_lexical_diversity(prediction: torch.Tensor, vocab: Vocabulary) -> int:
    words = set()
    for i in prediction.cpu().numpy():
        word = vocab.idx2word.get(i)
        if word == "<end>":
            return len(words)

        if word not in {"<start>", "<pad>", "<eoi>"}:
            words.add(word)
    return len(words)


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
