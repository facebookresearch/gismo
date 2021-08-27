# Copyright (c) Facebook, Inc. and its affiliates.
#
# Code adapted from https://github.com/facebookresearch/inversecooking
#
# This source code is licensed under the MIT license found in the
# LICENSE file in https://github.com/facebookresearch/inversecooking

import json
import os
import pickle
import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from inv_cooking.datasets.recipe1m.parsing import (
    IngredientParser,
    InstructionParser,
    cluster_ingredients,
    match_flavorgraph,
    remove_plurals,
    remove_plurals_flavorgraph,
)
from inv_cooking.datasets.recipe1m.parsing.titles import TitleParser
from inv_cooking.datasets.vocabulary import Vocabulary

BASE_WORDS = [
    "peppers",
    "tomato",
    "spinach_leaves",
    "turkey_breast",
    "lettuce_leaf",
    "chicken_thighs",
    "milk_powder",
    "bread_crumbs",
    "onion_flakes",
    "red_pepper",
    "pepper_flakes",
    "juice_concentrate",
    "cracker_crumbs",
    "hot_chili",
    "seasoning_mix",
    "dill_weed",
    "pepper_sauce",
    "sprouts",
    "cooking_spray",
    "cheese_blend",
    "basil_leaves",
    "pineapple_chunks",
    "marshmallow",
    "chile_powder",
    "cheese_blend",
    "corn_kernels",
    "tomato_sauce",
    "chickens",
    "cracker_crust",
    "lemonade_concentrate",
    "red_chili",
    "mushroom_caps",
    "mushroom_cap",
    "breaded_chicken",
    "frozen_pineapple",
    "pineapple_chunks",
    "seasoning_mix",
    "seaweed",
    "onion_flakes",
    "bouillon_granules",
    "lettuce_leaf",
    "stuffing_mix",
    "parsley_flakes",
    "chicken_breast",
    "basil_leaves",
    "baguettes",
    "green_tea",
    "peanut_butter",
    "green_onion",
    "fresh_cilantro",
    "breaded_chicken",
    "hot_pepper",
    "dried_lavender",
    "white_chocolate",
    "dill_weed",
    "cake_mix",
    "cheese_spread",
    "turkey_breast",
    "chucken_thighs",
    "basil_leaves",
    "mandarin_orange",
    "laurel",
    "cabbage_head",
    "pistachio",
    "cheese_dip",
    "thyme_leave",
    "boneless_pork",
    "red_pepper",
    "onion_dip",
    "skinless_chicken",
    "dark_chocolate",
    "canned_corn",
    "muffin",
    "cracker_crust",
    "bread_crumbs",
    "frozen_broccoli",
    "philadelphia",
    "cracker_crust",
    "chicken_breast",
]


def update_counter(sentence_list, counter_toks):
    for sentence in sentence_list:
        tokens = nltk.tokenize.word_tokenize(sentence)
        counter_toks.update(tokens)


def build_vocab_recipe1m(
    dets, layer1, layer2, args: DictConfig, recipe1m_path: str = None
):
    id_to_images_index = {}
    for i, entry in enumerate(layer2):
        id_to_images_index[entry["id"]] = i

    # ingredient parser
    if args.flavor_graph:
        ingredient_parser = IngredientParser(
            replace_dict={"_": ["-"], "": ["#", "[", "]", "!", "?"]}
        )
    else:
        ingredient_parser = IngredientParser(
            replace_dict={
                "and": ["&", "'n"],
                "": ["%", ",", ".", "#", "[", "]", "!", "?"],
            }
        )
    # instruction parser
    instruction_parser = InstructionParser(
        replace_dict={"and": ["&", "'n"], "": ["#", "[", "]"]}
    )

    title_parser = TitleParser()

    idx2ind = {}
    for i, entry in enumerate(dets):
        idx2ind[entry["id"]] = i

    #####
    # 1. Count words in dataset and clean
    #####

    counter_ingrs = Counter()
    counter_recipe_tokens = Counter()
    counter_title_tokens = Counter()

    for i, entry in tqdm(enumerate(layer1)):

        # retrieve pre-detected ingredients for this entry
        det_entry = dets[idx2ind[entry["id"]]]
        ingrs_list = list(
            ingredient_parser.parse_entry(det_entry, clean_digits=not args.flavor_graph)
        )

        # get raw text for instructions of this entry
        acc_len, instrs_list = instruction_parser.parse_entry(entry)

        # discard recipes with too few or too many ingredients or instruction words
        if (
            len(ingrs_list) < args.minnumingrs
            or len(instrs_list) < args.minnuminstrs
            or len(instrs_list) >= args.maxnuminstrs
            or len(ingrs_list) >= args.maxnumingrs
            or acc_len < args.minnumwords
        ):
            continue

        # tokenize sentences and update counter
        if entry["partition"] == "train":
            update_counter(instrs_list, counter_recipe_tokens)
            title = title_parser.parse_entry(entry)
            counter_recipe_tokens.update(title)
            counter_title_tokens.update(title)
            counter_ingrs.update(ingrs_list)

    # manually add missing entries for better clustering
    for base_word in BASE_WORDS:
        if base_word not in counter_ingrs.keys():
            counter_ingrs[base_word] = 1

    if args.flavor_graph:
        ingrs_df = pd.read_csv(
            os.path.join(
                os.path.dirname(recipe1m_path), "flavorgraph", "nodes_191120.csv"
            )
        )
        ingredients = ingrs_df.loc[ingrs_df["node_type"] == "ingredient"]
        all_flavorgraph_ingrs = ingredients["name"]
        pattern = "(?P<char>[" + re.escape("_") + "])(?P=char)+"
        counter_ingrs = {
            k.replace(k, re.sub(pattern, r"\1", k)): v for k, v in counter_ingrs.items()
        }
        cluster_ingrs, counter_ingrs = remove_plurals_flavorgraph(counter_ingrs)
        cluster_ingrs, counter_ingrs = match_flavorgraph(
            counter_ingrs, cluster_ingrs, list(all_flavorgraph_ingrs), recipe1m_path
        )

        # If the ingredient frequency is less than 'threshold', then the ingredient is discarded.
        words = [
            word
            for word, cnt in counter_recipe_tokens.items()
            if cnt >= args.threshold_words
        ]
        ingrs = cluster_ingrs
    else:
        counter_ingrs, cluster_ingrs = cluster_ingredients(counter_ingrs)
        counter_ingrs, cluster_ingrs = remove_plurals(counter_ingrs, cluster_ingrs)

        # If the ingredient frequency is less than 'threshold', then the ingredient is discarded.
        words = [
            word
            for word, cnt in counter_recipe_tokens.items()
            if cnt >= args.threshold_words
        ]
        ingrs = {
            word: cnt
            for word, cnt in counter_ingrs.items()
            if cnt >= args.threshold_ingrs
        }
    title_words = [
        word
        for word, cnt in counter_title_tokens.items()
        if cnt >= args.threshold_title
    ]

    # Recipe vocab
    # Create a vocab wrapper and add some special tokens.
    vocab_recipe = Vocabulary()
    vocab_recipe.add_word("<start>")
    vocab_recipe.add_word("<end>")
    vocab_recipe.add_word("<eoi>")
    for word in words:
        vocab_recipe.add_word(word)
    vocab_recipe.add_word("<pad>")

    # Create a vocabulary for titles
    # - The use case is prediction of titles as potential pre-training
    vocab_title = Vocabulary()
    vocab_title.add_word("<end>")
    for word in title_words:
        vocab_title.add_word(word)
    vocab_title.add_word("<pad>")

    # Ingredient vocab
    # Create a vocab wrapper for ingredients
    vocab_ingrs = Vocabulary()
    vocab_ingrs.add_word("<end>")
    for k, _ in ingrs.items():
        vocab_ingrs.add_word_group(cluster_ingrs[k])
    vocab_ingrs.add_word("<pad>")

    print("Total ingr vocabulary size: {}".format(len(vocab_ingrs)))
    print("Total recipe token vocabulary size: {}".format(len(vocab_recipe)))
    print("Total title token vocabulary size: {}".format(len(vocab_title)))

    dataset = {"train": [], "val": [], "test": []}

    ######
    # 2. Tokenize and build dataset based on vocabularies.
    ######
    for i, entry in tqdm(enumerate(layer1)):

        # retrieve pre-detected ingredients for this entry
        labels = []
        ingrs_list = []
        det_entry = dets[idx2ind[entry["id"]]]
        for det_ingr_undrs in ingredient_parser.parse_entry(
            det_entry, clean_digits=not args.flavor_graph
        ):
            ingrs_list.append(det_ingr_undrs)
            label_idx = vocab_ingrs(det_ingr_undrs)
            if label_idx is not vocab_ingrs("<pad>") and label_idx not in labels:
                labels.append(label_idx)

        # get raw text for instructions of this entry
        acc_len, instrs_list = instruction_parser.parse_entry(entry)

        # we discard recipes with too many or too few ingredients or instruction words
        if (
            len(labels) < args.minnumingrs
            or len(instrs_list) < args.minnuminstrs
            or len(instrs_list) >= args.maxnuminstrs
            or len(labels) >= args.maxnumingrs
            or acc_len < args.minnumwords
        ):
            continue

        # copy image paths for this recipe
        images_list = []
        if entry["id"] in id_to_images_index.keys():
            image_entry = layer2[id_to_images_index[entry["id"]]]
            for image in image_entry["images"]:
                images_list.append(image["id"])

        # tokenize sentences
        tokenized_instructions = []
        for instr in instrs_list:
            tokens = nltk.tokenize.word_tokenize(instr)
            tokenized_instructions.append(tokens)

        # tokenize title
        title = title_parser.parse_entry(entry)

        new_entry = {
            "id": entry["id"],
            "instructions": instrs_list,
            "tokenized": tokenized_instructions,
            "ingredients": ingrs_list,
            "images": images_list,
            "title": title,
        }
        dataset[entry["partition"]].append(new_entry)

    print("Dataset size:")
    for split in dataset.keys():
        print(split, ":", len(dataset[split]))

    return vocab_ingrs, vocab_recipe, vocab_title, dataset


def load_unprocessed_dataset(recipe1m_path):
    print("Loading data...")
    dets = json.load(open(os.path.join(recipe1m_path, "det_ingrs.json"), "r"))
    layer1 = json.load(open(os.path.join(recipe1m_path, "layer1.json"), "r"))
    layer2 = json.load(open(os.path.join(recipe1m_path, "layer2.json"), "r"))
    print("Loaded data.")
    print(f"Found {len(layer1)} recipes in the dataset.")
    return dets, layer1, layer2


def save_vocabulary(folder: str, file: str, vocab: Vocabulary):
    ingredients_path = os.path.join(folder, file)
    with open(ingredients_path, "wb") as f:
        pickle.dump(vocab, f)


def run_dataset_pre_processing(recipe1m_path: str, config: DictConfig):

    train_split_path = Path(os.path.join(config.save_path, "final_recipe1m_train.pkl"))
    val_split_path = Path(os.path.join(config.save_path, "final_recipe1m_val.pkl"))
    test_split_path = Path(os.path.join(config.save_path, "final_recipe1m_test.pkl"))
    vocab_ingrs_path = Path(
        os.path.join(config.save_path, "final_recipe1m_vocab_ingrs.pkl")
    )

    if (
        train_split_path.exists()
        and val_split_path.exists()
        and test_split_path.exists()
        and vocab_ingrs_path.exists()
    ):
        dataset = {}
        with train_split_path.open("rb") as handle:
            dataset["train"] = pickle.load(handle)
        with val_split_path.open("rb") as handle:
            dataset["val"] = pickle.load(handle)
        with test_split_path.open("rb") as handle:
            dataset["test"] = pickle.load(handle)
        with vocab_ingrs_path.open("rb") as handle:
            vocab_ingrs = pickle.load(handle)
        return vocab_ingrs, dataset

    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)

    # Load the data files of Recipe1M
    dets, layer1, layer2 = load_unprocessed_dataset(recipe1m_path)

    # Build vocabularies and dataset
    vocab_ingrs, vocab_toks, vocab_title, dataset = build_vocab_recipe1m(
        dets,
        layer1,
        layer2,
        config,
        recipe1m_path,
    )

    # Save the vocabularies
    save_vocabulary(config.save_path, "final_recipe1m_vocab_ingrs.pkl", vocab_ingrs)
    save_vocabulary(config.save_path, "final_recipe1m_vocab_toks.pkl", vocab_toks)
    save_vocabulary(config.save_path, "final_recipe1m_vocab_title.pkl", vocab_title)

    # Save the dataset
    for split in dataset.keys():
        split_file_name = "final_recipe1m_" + split + ".pkl"
        split_file_name = os.path.join(config.save_path, split_file_name)
        with open(split_file_name, "wb") as f:
            pickle.dump(dataset[split], f)

    return vocab_ingrs, dataset
