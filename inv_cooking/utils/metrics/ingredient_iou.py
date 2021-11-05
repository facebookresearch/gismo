import bisect
from collections import deque
from typing import List, Set

import torch

from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.utils.metrics import DistributedAverage


class TrieIngredientExtractor:
    """
    Extraction of ingredients from a generated recipe:
    - perfect match of ingredients are preferred
    - then makes use of a TRIE data structure to find imperfect matches

    The algorithm consists in a sliding window, moving through the
    generated text, allowing to match ingredients from the end.

    Why the end? Because in english the most important word is the
    last one, previous one are only qualifiers (heuristic).

    The trie data structure allows to find the imperfect match by
    filtering on the prefix.
    """

    def __init__(self, ingr_vocab: Vocabulary):
        super().__init__()
        self.ingr_vocab = ingr_vocab
        self._build_trie()

    def _build_trie(self):
        self.max_window = 0
        self.valid_words = set()
        self.ingr_keys = []
        self.ingr_ids = []
        for ingr, ingr_id in self.ingr_vocab.word2idx.items():
            if ingr == "<pad>":
                continue

            # most important word at the end in english
            words = ingr.split("_")
            key = "_".join(words[::-1])
            self.max_window = max(self.max_window, len(words))
            self.ingr_keys.append(key)
            self.ingr_ids.append(ingr_id)
            for word in words:
                self.valid_words.add(word)

        # Sort the values to create a trie
        sort_indices = list(range(len(self.ingr_keys)))
        sort_indices.sort(key=lambda i: self.ingr_keys[i])
        self.ingr_keys = [self.ingr_keys[i] for i in sort_indices]
        self.ingr_ids = [self.ingr_ids[i] for i in sort_indices]

    def _filter_recipe(self, words: List[str]):
        return [word for word in words if word in self.valid_words]

    def find_ingredients(self, recipe_words: List[str], preferred_ingredients: Set[int]):
        recipe_words = self._filter_recipe(recipe_words)
        if len(recipe_words) < self.max_window:
            return set()

        # To identify already found ingredients and already used words
        # this allows to eliminate imperfect match if a perfect match
        # has already been found
        found_ingredients = set()
        consumed_words = [False] * len(recipe_words)

        # Going from biggest window to smallest to find more perfect
        # matches first
        for window_size in reversed(range(1, self.max_window + 1)):
            if window_size > len(recipe_words):
                continue

            # Moving a sliding window from left to right through text
            window = deque(range(-1, window_size - 1))
            remaining = deque(range(window_size - 1, len(recipe_words)))
            while remaining:
                window.popleft()
                window.append(remaining.popleft())
                if any(consumed_words[i] for i in window):
                    continue

                # Search the combination of words in the trie
                window_words = "_".join(recipe_words[i] for i in reversed(window))
                lo = bisect.bisect_left(self.ingr_keys, window_words)

                # No match possible
                if lo >= len(self.ingr_keys):
                    continue

                # Perfect match found:
                # Only done if window size is above 2, or else things
                # like "oil" might refer to "vegetable oil", we want to
                # avoid punishing for using abbrevatiations
                if window_size > 1 and self.ingr_keys[lo] == window_words:
                    found_ingredients.add(self.ingr_ids[lo])
                    for i in window:
                        consumed_words[i] = True

                # Imperfect match:
                # - try all ingredients that have the same prefix
                # - take the most specific one from preferred list
                backups = []
                backup_selected = False
                curr = lo
                while curr < len(self.ingr_keys) and self.ingr_keys[curr].startswith(window_words):
                    backups.append(self.ingr_ids[curr])
                    if self.ingr_ids[curr] in preferred_ingredients:
                        found_ingredients.add(self.ingr_ids[curr])
                        backup_selected = True
                        for i in window:
                            consumed_words[i] = True
                    curr += 1

                # Backup perfect match in case of window of size 1
                if not backup_selected:
                    if self.ingr_keys[lo] == window_words:
                        found_ingredients.add(self.ingr_ids[lo])
                        for i in window:
                            consumed_words[i] = True

        return found_ingredients

    def viz_ingredients(self, recipe_words: List[str], preferred_ingredients: Set[int]):
        found = self.find_ingredients(recipe_words, preferred_ingredients)
        fount_txt = sorted(self.ingr_vocab.idx2word[f][0] for f in found)
        print(found, fount_txt)
        return found, fount_txt


class IngredientIoU(DistributedAverage):
    """
    Metric to compute how well the ingredients appear in the recipe
    """

    def __init__(self, ingr_vocab: Vocabulary, instr_vocab: Vocabulary):
        super().__init__()
        self.ingr_vocab = ingr_vocab
        self.instr_vocab = instr_vocab
        self.ingredient_extractor = TrieIngredientExtractor(ingr_vocab=self.ingr_vocab)

    def add(self, ingredients: torch.tensor, recipes: torch.Tensor):
        # TODO - make sure those are the ingredients deduced!
        results = []
        batch_size = recipes.shape[0]
        for i in range(batch_size):
            iou, found, target = self.compute_iou(ingredients[i], recipes[i])
            results.append(iou)
        results = torch.tensor(results, device=ingredients.device)
        self.update(results)
        return results

    def compute_iou(self, ingredients: torch.Tensor, recipe: torch.Tensor):

        # Identify which ingredients appear in the recipe
        target_ingredients = self.filter_ingredients(ingredients)
        recipe_words = self.recipe_to_words(recipe, self.instr_vocab)
        found_ingredients = self.ingredient_extractor.find_ingredients(
            recipe_words=recipe_words,
            preferred_ingredients=target_ingredients,
        )

        # Compute the Intersection Over Union
        intersection = len(found_ingredients & target_ingredients)
        union = len(found_ingredients | target_ingredients)
        if union > 0:
            return intersection / union, found_ingredients, target_ingredients
        else:
            return 0.0, found_ingredients, target_ingredients

    def visusalise_iou(self, ingredients: torch.Tensor, recipe: torch.Tensor):
        iou, found, target = self.compute_iou(ingredients, recipe)
        print("IoU:", iou)
        print("Found:", sorted(self.ingr_vocab.idx2word[f][0] for f in found))
        print("Targets:", sorted(self.ingr_vocab.idx2word[t][0] for t in target))

    def filter_ingredients(self, ingredients: torch.Tensor) -> Set[int]:
        filtered_ingredients = set()
        for i in ingredients.cpu().numpy():
            words = self.ingr_vocab.idx2word[i]
            if words != "<pad>" and words != "<end>":
                filtered_ingredients.add(i)
        return filtered_ingredients

    @staticmethod
    def recipe_to_words(prediction: torch.Tensor, vocab: Vocabulary) -> List[str]:
        words = []
        for i in prediction.cpu().numpy():
            word = vocab.idx2word.get(i)
            if word == "<end>":
                return words
            if word not in {"<eoi>", "<start>", "<pad>"}:
                words.append(word)
        return words
