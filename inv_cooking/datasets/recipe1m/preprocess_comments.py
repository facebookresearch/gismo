"""
This code is adapted from:
https://github.com/ChantalMP/Exploiting-Food-Embeddings-for-Ingredient-Substitution/blob/master/relation_extraction/prepare_dataset.py
"""
import json
import os
from pathlib import Path

import numpy as np
import spacy
from tqdm import tqdm

from inv_cooking.datasets.recipe1m.substitution_dataset_generator import (
    create_substitutions_datasets,
)
from inv_cooking.datasets.recipe1m.normalisation.helpers.recipe_normalizer import (
    RecipeNormalizer,
)
from inv_cooking.datasets.recipe1m.normalisation.normalize_recipe_instructions import (
    normalize_instruction,
)


def split_reviews_to_sentences(all_reviews, all_ids):
    sentence_splitter = spacy.load("en_core_web_lg", disable=["tagger", "ner"])
    all_sentences = []
    all_ids_sentences = []
    chunksize = 10000
    chunked_data = [
        all_reviews[x : x + chunksize] for x in range(0, len(all_reviews), chunksize)
    ]
    chunked_id = [all_ids[x : x + chunksize] for x in range(0, len(all_ids), chunksize)]
    for ind, chunk in enumerate(
        tqdm(chunked_data, desc="Splitting Reviews to Sentences")
    ):
        reviews_docs = list(
            sentence_splitter.pipe(chunk, n_process=-1, batch_size=1000)
        )
        for ind2, review_doc in enumerate(reviews_docs):
            curr_sentences = [elem.text for elem in review_doc.sents]
            all_sentences.extend(curr_sentences)
            all_ids_sentences.extend(
                np.repeat(chunked_id[ind][ind2], len(curr_sentences))
            )

    return all_sentences, all_ids_sentences


def normalize_reviews(all_sentences, all_sentences_ids, flavorgraph_ingredients):

    ingredients_yummly_set = {tuple(ing.split(" ")) for ing in flavorgraph_ingredients}
    review_normalizer = RecipeNormalizer()
    normalized_reviews_docs = review_normalizer.model.pipe(
        all_sentences, n_process=-1, batch_size=1000
    )
    normalized_reviews = []
    normalized_reviews_dict = {}
    normalized_ids = []

    for ind, normalized_review_doc in enumerate(
        tqdm(normalized_reviews_docs, desc="Normalizing Reviews")
    ):
        normalized_review = normalize_instruction(
            normalized_review_doc,
            ingredients_yummly_set,
            instruction_normalizer=review_normalizer,
        )
        repeated_review = False
        if normalized_review in normalized_reviews_dict:
            if normalized_reviews_dict[normalized_review] == all_sentences_ids[ind]:
                repeated_review = True
        if not repeated_review:
            normalized_reviews.append(normalized_review)
            normalized_ids.append(all_sentences_ids[ind])
            normalized_reviews_dict[normalized_review] = all_sentences_ids[ind]
    return normalized_reviews, normalized_ids

# find the shortest distance of ingredients to substitution key terms
def find_distance_to_substitution(text, ingredient_words, substitution_terms):
    text = text.strip()
    indices_subs = []
    indices_ing = []
    for term in substitution_terms:
        if term in text:
            res = [i for i in range(len(text)) if text.startswith(term, i)]
            indices_subs += res
    for ing in ingredient_words:
        res = [i for i in range(len(text)) if text.startswith(ing, i)]
        indices_ing += res
    for ind1 in indices_subs:
        for ind2 in indices_ing:
            if len(text[min(ind1, ind2) : max(ind1, ind2)].split()) < 7:
                return True
    return False


def convert_reviews_to_dataset(normalized_reviews, normalized_ids, ingredients, mode):

    review_examples = []

    substitution_terms = ["instead", "substitute", "in place of", "replace"]
    ings_with_subs = [
        "brown_sugar_substitute",
        "butter_substitute",
        "egg_substitute",
        "ener_g_egg_substitute",
        "equal_sugar_substitute",
        "fat_free_egg_substitute",
        "liquid_egg_substitute",
        "non_dairy_milk_substitute",
        "salt_substitute",
        "splenda_sugar_substitute",
        "sugar_substitute",
        "xylitol_sugar_substitute",
    ]

    for ind, review in enumerate(
        tqdm(
            normalized_reviews,
            total=len(normalized_reviews),
            desc="Processing Normalized Reviews",
        )
    ):
        substitution_context = False
        for term in substitution_terms:
            if term in review:
                substitution_context = True
        if substitution_context:
            cleaned_review = (
                review.replace("£", "")
                .replace("$", "")
                .replace("!", " !")
                .replace("?", " ?")
                .replace(".", " .")
                .replace(":", " :")
                .replace(",", " ,")
                .replace(";", " ;")
            )
            cleaned_review = " " + cleaned_review + " "

            if mode == 1:
                sub_reviews = [cleaned_review]
            elif mode > 1:
                if mode == 2:
                    cleaned_review_wo_and = cleaned_review
                elif mode == 3:
                    cleaned_review_wo_and = cleaned_review.replace("and", ",")
                sub_reviews = cleaned_review_wo_and.split(",")

            for sub_review in sub_reviews:
                words = sub_review.split()
                ingredient_words = [word for word in words if word in ingredients]
                substitution_context = False
                temp_words = []
                for w in words:
                    if w not in ings_with_subs:
                        temp_words.append(w)
                temp_review = " ".join(temp_words)
                for term in substitution_terms:
                    if term in temp_review:
                        substitution_context = True

                ingredient_words = list(set(ingredient_words))
                close_enough = find_distance_to_substitution(
                    cleaned_review, ingredient_words, substitution_terms
                )

                if (
                    len(set(ingredient_words)) == 2
                    and substitution_context
                    and close_enough
                ):
                    cleaned_review_sub = cleaned_review.replace(
                        f" {ingredient_words[0]} ", f" $ {ingredient_words[0]} $ ", 1
                    )
                    text = cleaned_review_sub.replace(
                        f" {ingredient_words[1]} ", f" £ {ingredient_words[1]} £ ", 1
                    )

                    example = {
                        "id": normalized_ids[ind],
                        "text": text.strip(),
                    }
                    review_examples.append(example)
    return review_examples


def load_splits(dataset):
    split_ids = {"train": {}, "val": {}, "test": {}}
    for split in split_ids.keys():
        for recipe in dataset[split]:
            split_ids[split][recipe["id"]] = True
    return split_ids["train"], split_ids["val"], split_ids["test"]


def read_ingredients(vocab_ingrs):
    ingredients = vocab_ingrs.word2idx
    all_ingredients = []
    cleaned_ingredients = []
    for ing in ingredients:
        all_ingredients.append(ing)
        cleaned_ingredients.append(ing.replace("_", " "))
    return all_ingredients, cleaned_ingredients


def load_split_per_dataset(dataset, train_ids, val_ids, test_ids):
    train_dataset = []
    val_dataset = []
    test_dataset = []

    print("Splitting the dataset")
    for elem in dataset:
        id_ = elem["id"]
        if id_ in train_ids:
            train_dataset.append(elem)
        elif id_ in val_ids:
            val_dataset.append(elem)
        elif id_ in test_ids:
            test_dataset.append(elem)
        else:
            print("data not in any split")

    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print(
        f"Length of train: {len(train_dataset)}, val: {len(val_dataset)} and test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def run_comment_pre_processing(recipe1m_path, preprocessed_dir, vocab_ingrs, splits):
    review_sentences_path = Path(
        os.path.join(preprocessed_dir, "review_sentences_context.json")
    )
    review_ids_sentences_path = Path(
        os.path.join(preprocessed_dir, "review_ids_sentences_context.json")
    )
    normalized_review_sentences_path = Path(
        os.path.join(preprocessed_dir, "review_sentences_normalized_context.json")
    )
    normalized_review_ids_sentences_path = Path(
        os.path.join(preprocessed_dir, "review_ids_sentences_normalized_context.json")
    )

    comment_file_name = "recipe1m_with_reviews.json"

    all_ingredients, cleaned_ingredients = read_ingredients(vocab_ingrs)

    if review_sentences_path.exists() and review_ids_sentences_path.exists():
        with review_sentences_path.open() as f:
            all_sentences = json.load(f)
        with review_ids_sentences_path.open() as f:
            all_sentences_ids = json.load(f)
    else:
        all_reviews, all_ids = [], []
        file_path = recipe1m_path / comment_file_name
        with file_path.open() as f:
            recipes = json.load(f)
        for recipe in recipes:
            if "reviews" in recipe:
                all_reviews.extend(recipe["reviews"])
                all_ids.extend(np.repeat(recipe["id"], len(recipe["reviews"])))
        print(
            f"All reviews: {len(all_reviews)} and {len(all_ids)} after {comment_file_name}"
        )

        all_sentences, all_sentences_ids = split_reviews_to_sentences(
            all_reviews, all_ids
        )
        with review_sentences_path.open("w") as f:
            json.dump(all_sentences, f)
        with review_ids_sentences_path.open("w") as f:
            json.dump(all_sentences_ids, f)

    if (
        normalized_review_sentences_path.exists()
        and normalized_review_ids_sentences_path.exists()
    ):
        with normalized_review_sentences_path.open() as f:
            normalized_reviews = json.load(f)
        with normalized_review_ids_sentences_path.open() as f:
            normalized_ids = json.load(f)
    else:
        normalized_reviews, normalized_ids = normalize_reviews(
            all_sentences, all_sentences_ids, cleaned_ingredients
        )
        with normalized_review_sentences_path.open("w") as f:
            json.dump(normalized_reviews, f)
        with normalized_review_ids_sentences_path.open("w") as f:
            json.dump(normalized_ids, f)

        print("Normalized reviews:", len(normalized_reviews), len(normalized_ids))

    train_ids, val_ids, test_ids = load_splits(splits)

    dataset1 = convert_reviews_to_dataset(
        normalized_reviews, normalized_ids, all_ingredients, mode=1
    )
    dataset2 = convert_reviews_to_dataset(
        normalized_reviews, normalized_ids, all_ingredients, mode=2
    )
    dataset3 = convert_reviews_to_dataset(
        normalized_reviews, normalized_ids, all_ingredients, mode=3
    )
    dataset = dataset1 + dataset2 + dataset3

    train_dataset = []
    val_dataset = []
    test_dataset = []

    print("Splitting the dataset")
    for elem in dataset:
        id_ = elem["id"]
        if id_ in train_ids:
            train_dataset.append(elem)
        elif id_ in val_ids:
            val_dataset.append(elem)
        elif id_ in test_ids:
            test_dataset.append(elem)

    print(
        f"Length of train: {len(train_dataset)}, val: {len(val_dataset)} and test: {len(test_dataset)}"
    )

    create_substitutions_datasets(
        train_dataset, val_dataset, test_dataset, recipe1m_path, vocab_ingrs, splits
    )
