import json
import pickle
import random

import numpy as np
from metrics import Metrics


def context_free_examples(examples, vocabs, mode=0):
    output = []
    for example in examples:
        subs = example["subs"]
        subs = vocabs.word2idx[subs[0]], vocabs.word2idx[subs[1]]
        if mode == 0:
            output.append(subs)
        elif mode == 1:
            if subs not in output:
                output.append(subs)
    return output


def load_split_data(split):
    examples = json.load(
        open("/private/home/baharef/inversecooking2.0/preprocessed_data/" + split + "_comments_subs_deduplicated.pkl", "r")
    )
    return examples


def load_dict(subs, vocabs):
    subs_dict = {}

    for ing in subs:
        ing_id = vocabs.word2idx[ing]
        subs_list = []
        for ing_subs in subs[ing]:
            subs_list.append(vocabs.word2idx[ing_subs.replace(" ", "_")])
        subs_dict[ing_id] = subs_list
    return subs_dict


def load_vocab():
    vocab_ing = pickle.load(
        open("../preprocessed_data/final_recipe1m_vocab_ingrs.pkl", "rb")
    )
    return vocab_ing


def test_model(model_name, trial, split="test"):
    output_file = open('outputs/' + model_name + "test_ranks_i" + str(trial) + '.txt', 'w')
    metrics = Metrics()
    vocabs = load_vocab()
    examples = load_split_data(split)
    examples_cf = context_free_examples(examples, vocabs)

    if model_name == "food2vec":
        subs = json.load(
            open(
                "/private/home/baharef/Exploiting-Food-Embeddings-for-Ingredient-Substitution/"
                + model_name
                + "/data/substitute_pairs_food2vec_text_without_filtering6612.json",
                "r",
            )
        )
    elif model_name == "foodbert":
        subs = json.load(
            open(
                "/private/home/baharef/Exploiting-Food-Embeddings-for-Ingredient-Substitution/foodbert_embeddings/data/substitute_pairs_foodbert_without_filtering.json",
                "r",
            )
        )
    
    elif model_name == "rbert":
        subs_list = json.load(
            open(
                "/private/home/baharef/Exploiting-Food-Embeddings-for-Ingredient-Substitution/relation_extraction/data/substitute_pairs_relation_extraction.json",
                "r",
            )
        )
        subs = {}
        for example in subs_list:
            example[0] = example[0].replace(' ', '_')
            example[1] = example[1].replace(' ', '_')
            if example[0] not in subs:
                subs[example[0]] = []
            subs[example[0]].append(example[1])

    # print("dictionary loaded!")

    subs_dict = load_dict(subs, vocabs)
    for example in examples_cf:
        try:
            subs = subs_dict[example[0]]
            rank = subs.index(example[1]) + 1
        except Exception:
            rank = random.randint(0, 6633) + 1
        metrics.update(rank)

        output_file.write(str(example[0]) + " " + str(example[1]) + " " + str(rank) + "\n")
    return metrics.normalize()


if __name__ == "__main__":
    model_name = "food2vec"
    print("===========", model_name, "===========")
    mrrs = []
    hits1 = []
    hits3 = []
    hits10 = []
    for trial in range(5):
        mrr, hit1, hit3, hit10 = test_model(model_name, trial)
        mrrs.append(mrr)
        hits1.append(hit1)
        hits3.append(hit3)
        hits10.append(hit10)
    print("mrr:", np.mean(mrrs), np.std(mrrs))
    print("hit@1:", np.mean(hits1), np.std(hits1))
    print("hit@3:", np.mean(hits3), np.std(hits3))
    print("hit@10:", np.mean(hits10), np.std(hits10))
