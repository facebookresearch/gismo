"""
Used to generate cleaned_recipe1m.json from recipe1m.json by cleaning the instructions
This includes lemmatization, merging multi word ingredients with underscore etc.
"""
import re

from inv_cooking.datasets.recipe1m.normalisation.helpers.recipe_normalizer import (
    RecipeNormalizer,
)


def match_ingredients(normalized_instruction_tokens, yummly_ingredients_set, n):
    not_word_tokens = [".", ",", "!", "?", " ", ";", ":"]
    for i in range(len(normalized_instruction_tokens) - n, -1, -1):
        sublist = normalized_instruction_tokens[i : i + n]
        if sublist[0] in not_word_tokens or sublist[-1] in not_word_tokens:
            continue
        clean_sublist = tuple(
            [token for token in sublist if token not in not_word_tokens]
        )
        if clean_sublist in yummly_ingredients_set:
            new_instruction_tokens = []
            new_ingredient = "_".join(clean_sublist)
            for idx, token in enumerate(normalized_instruction_tokens):
                if idx < i or idx >= i + n:
                    new_instruction_tokens.append(token)
                elif new_ingredient is not None:
                    new_instruction_tokens.append(new_ingredient)
                    new_ingredient = None
            return new_instruction_tokens, True
    return normalized_instruction_tokens, False


def normalize_instruction(
    instruction_doc, yummly_ingredients_set, instruction_normalizer: RecipeNormalizer
):
    normalized_instruction = ""
    for idx, word in enumerate(instruction_doc):
        if not word.is_punct:  # we want a space before all non-punctuation words
            space = " "
        else:
            space = ""
        if word.tag_ in ["NN", "NNS", "NNP", "NOUN", "NNPS"]:
            normalized_instruction += (
                space
                + instruction_normalizer.lemmatize_token_to_str(
                    token=word, token_tag="NOUN"
                )
            )
        else:
            normalized_instruction += space + word.text

    normalized_instruction = normalized_instruction.strip()

    normalized_instruction_tokens = re.findall(
        r"[\w'-]+|[.,!?; ]", normalized_instruction
    )
    # find all sublists of tokens with descending length
    for n in range(
        8, 1, -1
    ):  # stop at 2 because matching tokens with length 1 can stay as they are
        match = True
        while match:
            normalized_instruction_tokens, match = match_ingredients(
                normalized_instruction_tokens, yummly_ingredients_set, n
            )

    return "".join(normalized_instruction_tokens)
