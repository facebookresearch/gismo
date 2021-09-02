"""
    This repo is copied from:
    https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/tree/master/normalisation
"""

from typing import List

import spacy
from spacy.tokens import Token

from inv_cooking.datasets.recipe1m.normalisation.helpers.utils import (
    has_number,
    words_to_remove,
)


def custom_removel_component(doc):
    in_paranthesis = False
    for token in doc:
        if token.text == "(":
            in_paranthesis = True

        if (
            not in_paranthesis
            and not token.is_digit
            and token.text not in words_to_remove
            and token.lemma_ not in words_to_remove
            and not has_number(token)
            and token.text[0] != "-"
        ):
            token._.to_keep = True

        if token.text == ")":
            in_paranthesis = False

    return doc


class RecipeNormalizer:
    """
    Applies both typical normalization and also cleanup according to hardcoded words
    Takes a list of ingredients, uses spacy tokenizer to create docs out of them, applies a .pipe for efficient processing.
    Then creates new ingredients from not eliminated tokens

    Usage Example:
    ingredient_normalizer = IngredientNormalizer()
    ingredient_normalizer.normalize_ingredients(['10 flour tortillas  hot'])
    out: flour tortilla hot
    """

    def __init__(self, lemmatization_types=None):
        self.model = spacy.load(
            "en_core_web_lg", disable=["parser", "ner"]
        )  # Disable parts of the pipeline that are not necessary
        Token.set_extension("to_keep", default=False)  # new attribute for Token
        self.model.add_pipe(
            custom_removel_component, "custom_removel_component"
        )  # new component for the pipeline
        self.tag_mapping = {
            "NN": "NOUN",
            "NNS": "NOUN",
            "NNP": "NOUN",
            "NNPS": "NOUN",
            ".": "NOUN",
            "JJS": "ADJ",
            "JJR": "ADJ",
            "VBD": "VERB",
            "VBG": "VERB",
            "VBN": "VERB",
            "VBZ": "VERB",
            "VBP": "VERB",
        }

        # if not None, only lemmatize types in this list
        self.lemmatization_types = lemmatization_types
        self.lemmatizer = self.model.vocab.morphology.lemmatizer

    def lemmatize_token_to_str(self, token, token_tag):
        if self.lemmatization_types is None or token_tag in self.lemmatization_types:
            lemmatized = self.lemmatizer(token.text.lower(), token_tag)[0]
        else:
            lemmatized = token.text.lower()

        return lemmatized

    def normalize_ingredients(self, ingredients: List[str], strict=True):
        ingredients = [
            ingredient.split(",")[0] for ingredient in ingredients
        ]  # Ignore after comma
        # Disable unnecessary parts of the pipeline, also run at once with pipe which is more efficient
        ingredients_docs = self.model.pipe(ingredients, n_process=-1, batch_size=1000)

        cleaned_ingredients = []
        for ingredient_doc in ingredients_docs:
            cleaned_ingredient = []
            for token in ingredient_doc:
                token_tag = token.tag_
                if token_tag in self.tag_mapping:
                    token_tag = self.tag_mapping[token_tag]
                if strict:
                    if len(token) > 1 and token._.to_keep:
                        lemmatized = self.lemmatize_token_to_str(token, token_tag)
                        cleaned_ingredient.append(lemmatized)
                else:
                    if len(token) > 1:
                        lemmatized = self.lemmatize_token_to_str(token, token_tag)
                        cleaned_ingredient.append(lemmatized)

            cleaned_ingredients.append(" ".join(cleaned_ingredient))

        return cleaned_ingredients
