# Copyright (c) Facebook, Inc. and its affiliates.
#
# Code adapted from https://github.com/facebookresearch/inversecooking
#
# This source code is licensed under the MIT license found in the
# LICENSE file in https://github.com/facebookresearch/inversecooking

import json
import os
import pickle
from collections import Counter

import nltk
from omegaconf import DictConfig
from tqdm import tqdm

from inv_cooking.datasets.recipe1m.parsing import (
    IngredientParser,
    InstructionParser,
    cluster_ingredients,
    remove_plurals,
)
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


def build_vocab_recipe1m(dets, layer1, layer2, args: DictConfig):
    id_to_images_index = {}
    for i, entry in enumerate(layer2):
        id_to_images_index[entry["id"]] = i

    ingredient_parser = IngredientParser(
        replace_dict={"and": ["&", "'n"], "": ["%", ",", ".", "#", "[", "]", "!", "?"]}
    )

    instruction_parser = InstructionParser(
        replace_dict={"and": ["&", "'n"], "": ["#", "[", "]"]}
    )

    idx2ind = {}
    for i, entry in enumerate(dets):
        idx2ind[entry["id"]] = i

    #####
    # 1. Count words in dataset and clean
    #####

    counter_ingrs = Counter()
    counter_toks = Counter()

    for i, entry in tqdm(enumerate(layer1)):

        # retrieve pre-detected ingredients for this entry
        det_entry = dets[idx2ind[entry["id"]]]
        ingrs_list = list(ingredient_parser.parse_entry(det_entry))

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
            update_counter(instrs_list, counter_toks)
            title = nltk.tokenize.word_tokenize(entry["title"].lower())
            counter_toks.update(title)
            counter_ingrs.update(ingrs_list)

    # manually add missing entries for better clustering
    for base_word in BASE_WORDS:
        if base_word not in counter_ingrs.keys():
            counter_ingrs[base_word] = 1

    counter_ingrs, cluster_ingrs = cluster_ingredients(counter_ingrs)
    counter_ingrs, cluster_ingrs = remove_plurals(counter_ingrs, cluster_ingrs)

    # If the ingredient frequency is less than 'threshold', then the ingredient is discarded.
    words = [word for word, cnt in counter_toks.items() if cnt >= args.threshold_words]
    ingrs = {
        word: cnt for word, cnt in counter_ingrs.items() if cnt >= args.threshold_ingrs
    }

    # Recipe vocab
    # Create a vocab wrapper and add some special tokens.
    vocab_toks = Vocabulary()
    vocab_toks.add_word("<start>")
    vocab_toks.add_word("<end>")
    vocab_toks.add_word("<eoi>")
    for word in words:
        vocab_toks.add_word(word)
    vocab_toks.add_word("<pad>")

    # Ingredient vocab
    # Create a vocab wrapper for ingredients
    vocab_ingrs = Vocabulary()
    vocab_ingrs.add_word("<end>")
    for k, _ in ingrs.items():
        vocab_ingrs.add_word_group(cluster_ingrs[k])
    vocab_ingrs.add_word("<pad>")

    print("Total ingr vocabulary size: {}".format(len(vocab_ingrs)))
    print("Total token vocabulary size: {}".format(len(vocab_toks)))

    dataset = {"train": [], "val": [], "test": []}

    ######
    # 2. Tokenize and build dataset based on vocabularies.
    ######
    for i, entry in tqdm(enumerate(layer1)):

        # retrieve pre-detected ingredients for this entry
        labels = []
        ingrs_list = []
        det_entry = dets[idx2ind[entry["id"]]]
        for det_ingr_undrs in ingredient_parser.parse_entry(det_entry):
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
        title = nltk.tokenize.word_tokenize(entry["title"].lower())

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

    return vocab_ingrs, vocab_toks, dataset


def load_unprocessed_dataset(recipe1m_path):
    print("Loading data...")
    dets = json.load(open(os.path.join(recipe1m_path, "det_ingrs.json"), "r"))
    layer1 = json.load(open(os.path.join(recipe1m_path, "layer1.json"), "r"))
    layer2 = json.load(open(os.path.join(recipe1m_path, "layer2.json"), "r"))
    print("Loaded data.")
    print(f"Found {len(layer1)} recipes in the dataset.")
    return dets, layer1, layer2


def run_dataset_pre_processing(recipe1m_path: str, config: DictConfig):
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)

    # Load the data files of Recipe1M
    dets, layer1, layer2 = load_unprocessed_dataset(recipe1m_path)

    # Build vocabularies and dataset
    vocab_ingrs, vocab_toks, dataset = build_vocab_recipe1m(
        dets, layer1, layer2, config
    )

    # Save the vocabularies and dataset
    ingredients_path = os.path.join(config.save_path, "final_recipe1m_vocab_ingrs.pkl")
    with open(ingredients_path, "wb") as f:
        pickle.dump(vocab_ingrs, f)

    vocab_tokens_path = os.path.join(config.save_path, "final_recipe1m_vocab_toks.pkl")
    with open(vocab_tokens_path, "wb") as f:
        pickle.dump(vocab_toks, f)

    for split in dataset.keys():
        split_file_name = "final_recipe1m_" + split + ".pkl"
        split_file_name = os.path.join(config.save_path, split_file_name)
        with open(split_file_name, "wb") as f:
            pickle.dump(dataset[split], f)
