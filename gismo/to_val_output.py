# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from data_loader import load_data
from state_loader import create_output_dir
import torch
import os
import pickle
import hydra


# this function get the counter and return the name of an ingredient
def cnt_to_id(cnt, count2id, id2name):
    return id2name[count2id[cnt]]



@hydra.main(config_path="conf", config_name="config")
def to_val_output(cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph, train_dataset, val_dataset, test_dataset, ingrs, node_count2id, node_id2name, node_id2count, recipe_id2counter, filter_ingredient = load_data(
        cfg.nr,
        cfg.max_context,
        cfg.add_self_loop,
        cfg.neg_sampling,
        cfg.data_augmentation,
        cfg.p_augmentation,
        cfg.filter,
        device,
        data_path=os.path.expanduser(cfg.data_path),
    )
    output_dir = create_output_dir(cfg)
    rank_path = os.path.join(output_dir, "val_ranks.txt")

    replacements = []
    with open(rank_path, "r") as f:
        for line in f.readlines():
            ingredients = [int(x) for x in line.split(" ")]
            ingr_a = cnt_to_id(ingredients[0], node_count2id, node_id2name)
            ingr_subs = [
                cnt_to_id(ingr, node_count2id, node_id2name)
                for ingr in ingredients[2:]
            ]
            replacements.append([ingr_a] + ingr_subs)

    rank_path_translated = os.path.join(output_dir, "val_ranks_out.pkl")
    with open(rank_path_translated, "wb") as f:
        pickle.dump(replacements, f)


if __name__ == "__main__":
    to_val_output()
