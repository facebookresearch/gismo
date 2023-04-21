# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Code adapted from https://github.com/facebookresearch/inversecooking
# This source code is licensed under the MIT license found in the
# LICENSE file in https://github.com/facebookresearch/inversecooking

from typing import List, Set

import torch
from PIL import Image
import numpy as np

from inv_cooking.datasets.vocabulary import Vocabulary


@torch.no_grad()
def tensor_to_image(tensor: torch.Tensor):
    sigma = torch.as_tensor(
        (0.229, 0.224, 0.225), dtype=tensor.dtype, device=tensor.device
    ).view(-1, 1, 1)
    mu = torch.as_tensor(
        (0.485, 0.456, 0.406), dtype=tensor.dtype, device=tensor.device
    ).view(-1, 1, 1)
    tensor = (tensor * sigma) + mu
    tensor = tensor.permute((1, 2, 0))
    array = tensor.cpu().detach().numpy()
    array = np.uint8(array * 255)
    return Image.fromarray(array, mode="RGB")


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


def ingredients_to_text(prediction: torch.Tensor, vocab: Vocabulary, full_list: bool = False) -> List[str]:
    ingredient_list = []
    for i in prediction.cpu().numpy():
        word = vocab.idx2word.get(i)
        if word != "<pad>":
            if isinstance(word, list):
                if full_list:
                    ingredient_list.append(word)
                else:
                    ingredient_list.append(word[0])
            else:
                ingredient_list.append(word)
    return ingredient_list


def split_ingredient_components(ingredients: List[str]) -> List[Set[str]]:
    components = []
    for ingredient in ingredients:
        components.append(set(ingredient.split("_")))
    return components
