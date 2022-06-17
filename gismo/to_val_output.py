from data_loader import load_data
import torch
import os
import pickle
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph, train_dataset, val_dataset, test_dataset, ingrs, node_count2id, node_id2name, node_id2count, recipe_id2counter, filter_ingredient = load_data(
    400,
    43,
    False,
    'regular',
    False,
    0.5,
    False,
    device,
    # TODO(config)
    dir_="/private/home/qduval/baharef/inversecooking2.0/inversecooking2.0/data/flavorgraph",
    # dir_="/private/home/baharef/inversecooking2.0/data/flavorgraph",
)


# this function get the counter and return the name of an ingredient
def cnt_to_id(cnt, count2id, id2name):
    return id2name[count2id[cnt]]


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, help="path where to find the file to parse")
args = parser.parse_args()

output_dir = args.path
rank_path = os.path.join(output_dir, "test_ranks_full.txt")

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

