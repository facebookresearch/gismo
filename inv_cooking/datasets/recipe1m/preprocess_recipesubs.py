# Copyright (c) Facebook, Inc. and its affiliates.

import os
import pickle
from pathlib import Path


def create_pre_processed_recipesubs_data(
    pre_processed_dir: str, recipe_dataset: dict, substitution_dataset: dict
):

    dataset = {"train": [], "val": [], "test": []}

    vocab_path = Path(pre_processed_dir) / "final_recipe1m_vocab_ingrs.pkl"
    vocab = pickle.load(vocab_path.open("rb"))

    for split in ["train", "val", "test"]:

        # load preprocessed recipe dataset
        recipe_dataset_split = recipe_dataset[split]

        # load substitutions dataset
        substitution_dataset_split = substitution_dataset[split]

        # get all recipe ids
        recipes_by_id = {recipe["id"]: recipe for recipe in recipe_dataset_split}

        # create recipesubs dataset
        visited_keys = set()
        for i, subs in enumerate(substitution_dataset_split):
            if i % 500 == 0:
                print(
                    f"Processing {i} out of {len(substitution_dataset_split)} samples."
                )

            recipe_entry = recipes_by_id.get(subs["id"])
            if recipe_entry is None:
                continue

            ingr_before, ingr_after = subs["subs"]
            key = (recipe_entry["id"], vocab(ingr_before), vocab(ingr_after))
            if key in visited_keys:
                continue

            visited_keys.add(key)
            new_entry = {
                "id": recipe_entry["id"],
                "instructions": recipe_entry["instructions"],
                "tokenized": recipe_entry["tokenized"],
                "ingredients": recipe_entry["ingredients"],
                "images": recipe_entry["images"],
                "title": recipe_entry["title"],
                "comment": subs["text"],
                "substitution": subs["subs"],
            }
            dataset[split].append(new_entry)

        print(f"Recipe {split} has {len(recipe_dataset_split)} samples.")
        print(f"Substitutions {split} has {len(substitution_dataset_split)} samples.")
        print(f"Final {split} has {len(dataset[split])} samples.")

    # Save the dataset
    for split in dataset.keys():
        split_file_name = "final_recipe1msubs_" + split + ".pkl"
        split_file_name = os.path.join(pre_processed_dir, split_file_name)
        with open(split_file_name, "wb") as f:
            pickle.dump(dataset[split], f)

    return dataset
