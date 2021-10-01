import json
import random

import numpy as np
import utils
from metrics import Metrics


def context_free_examples(examples, vocabs, mode=0, data_augmentation=0):
    output = []
    for example in examples:
        subs = example["subs"]
        subs = vocabs.word2idx[subs[0]], vocabs.word2idx[subs[1]]
        if mode == 0:
            output.append(subs)

            if data_augmentation:
                subs_inv = subs[::-1]
                output.append(subs_inv)
        elif mode == 1:
            if subs not in output:
                output.append(subs)
        

    return output


def load_split_data(split):
    examples = json.load(
        open("../preprocessed_data/" + split + "_comments_subs.pkl", "r")
    )
    return examples


def load_vocab():
    vocab_ing, _ = utils.get_vocabs()
    # vocab_ing = pickle.load(
    #     open("../preprocessed_data/final_recipe1m_vocab_ingrs.pkl", "rb")
    # )
    pad_id = vocab_ing.word2idx["<pad>"]
    all_ids = list(vocab_ing.idx2word.keys())
    all_ids.remove(pad_id)
    return vocab_ing, all_ids


def create_lookup_table(examples):
    train_table = {}
    for example in examples:
        ing1, ing2 = example
        if ing1 not in train_table:
            train_table[ing1] = []
        train_table[ing1].append(ing2)
    return train_table


def create_lookup_table_ingredient_frequency(examples, vocabs):
    train_table = {}
    for example in examples:
        ing1, ing2 = example
        if ing1 not in train_table:
            train_table[ing1] = {}
        if ing2 not in train_table[ing1]:
            train_table[ing1][ing2] = 0
        train_table[ing1][ing2] += 1

    for ing in train_table:
        sorted_list = list(
            dict(
                sorted(
                    train_table[ing].items(),
                    key=lambda kv: (kv[1], kv[0]),
                    reverse=True,
                )
            ).keys()
        )
        train_table[ing] = sorted_list

    return train_table


def create_lookup_table_mode(examples):
    frequencies = {}
    for example in examples:
        ing1, ing2 = example
        if ing2 not in frequencies:
            frequencies[ing2] = 0
        frequencies[ing2] += 1
    max_val = 0
    max_key = None
    for word in frequencies:
        if frequencies[word] > max_val:
            max_val = frequencies[word]
            max_key = word
    return max_key


def create_lookup_table_frequency(examples):
    frequencies = {}
    for example in examples:
        ing1, ing2 = example
        if ing2 not in frequencies:
            frequencies[ing2] = 0
        frequencies[ing2] += 1

    frequency_list = list(
        {
            k: v
            for k, v in sorted(
                frequencies.items(), key=lambda item: item[1], reverse=True
            )
        }.keys()
    )
    return frequency_list


def test_lookup_table(test_examples, train_examples, vocabs, all_ids):
    metric = Metrics()
    train_examples_cf = context_free_examples(train_examples, vocabs)
    lookup_table = create_lookup_table(train_examples_cf)
    test_examples_cf = context_free_examples(test_examples, vocabs)

    for example in test_examples_cf:
        subs = []
        ing1, ing2 = example
        if ing1 in lookup_table:
            subs = list(set(lookup_table[ing1]))
        random.shuffle(subs)
        rest_ids = all_ids.copy()
        for id in subs:
            rest_ids.remove(id)
        rest_ids.remove(ing1)
        random.shuffle(rest_ids)
        all_subs = subs + rest_ids
        rank = all_subs.index(ing2) + 1
        metric.update(rank)
    return metric.normalize()


def test_lookup_table_frequency(test_examples, train_examples, vocabs, all_ids, data_augmentation=0):
    metric = Metrics()
    train_examples_cf = context_free_examples(train_examples, vocabs, data_augmentation=data_augmentation)
    lookup_table = create_lookup_table_ingredient_frequency(train_examples_cf, vocabs)
    test_examples_cf = context_free_examples(test_examples, vocabs)

    for example in test_examples_cf:
        subs = []
        ing1, ing2 = example
        if ing1 in lookup_table:
            subs = list(lookup_table[ing1])
        rest_ids = all_ids.copy()
        for id in subs:
            rest_ids.remove(id)
        rest_ids.remove(ing1)
        random.shuffle(rest_ids)
        all_subs = subs + rest_ids
        rank = all_subs.index(ing2) + 1
        metric.update(rank)

    return metric.normalize()


def test_random(test_examples, vocabs, all_ids):
    metric = Metrics()
    test_examples_cf = context_free_examples(test_examples, vocabs)
    for example in test_examples_cf:
        ing1, ing2 = example
        rest_ids = all_ids.copy()
        rest_ids.remove(ing1)
        random.shuffle(rest_ids)
        rank = rest_ids.index(ing2) + 1
        metric.update(rank)
    return metric.normalize()


def test_mode(test_examples, train_examples, vocabs, all_ids):
    metric = Metrics()
    train_examples_cf = context_free_examples(train_examples, vocabs)
    max_key = create_lookup_table_mode(train_examples_cf)
    test_examples_cf = context_free_examples(test_examples, vocabs)
    for example in test_examples_cf:
        ing1, ing2 = example
        rest_ids = all_ids.copy()
        rest_ids.remove(max_key)
        random.shuffle(rest_ids)
        all_subs = [max_key] + rest_ids
        all_subs.remove(ing1)
        rank = all_subs.index(ing2) + 1
        metric.update(rank)
    return metric.normalize()


def test_frequency(test_examples, train_examples, vocabs, all_ids):
    metric = Metrics()
    train_examples_cf = context_free_examples(train_examples, vocabs)
    freq_list = create_lookup_table_frequency(train_examples_cf)
    test_examples_cf = context_free_examples(test_examples, vocabs)

    for example in test_examples_cf:
        ing1, ing2 = example
        rest_ids = all_ids.copy()
        for id in freq_list:
            rest_ids.remove(id)
        random.shuffle(rest_ids)
        all_subs = freq_list.copy() + rest_ids
        all_subs.remove(ing1)
        rank = all_subs.index(ing2) + 1
        metric.update(rank)
    return metric.normalize()


def find_hard_examples(test_examples, train_examples):
    vocabs, all_ids = load_vocab()

    train_examples_cf = context_free_examples(train_examples, vocabs)
    test_examples_cf = context_free_examples(test_examples, vocabs)
    train_table = {}
    for example in train_examples_cf:
        ing1, ing2 = example
        if ing1 not in train_table:
            train_table[ing1] = {}
        if ing2 not in train_table[ing1]:
            train_table[ing1][ing2] = 0
        train_table[ing1][ing2] += 1

    train_table_frequent = {}
    for ing1 in train_table:
        for ing2 in train_table[ing1]:
            if train_table[ing1][ing2] >= 20:
                if ing1 not in train_table_frequent:
                    train_table_frequent[ing1] = []
                train_table_frequent[ing1].append(ing2)

    counter = 0
    for example in test_examples_cf:
        ing1, ing2 = example
        if ing1 in train_table_frequent and ing2 in train_table_frequent[ing1]:
            counter += 1
    print(counter / len(test_examples_cf))


if __name__ == "__main__":

    data_augmentation = 0
    train_examples = load_split_data("train")
    test_examples = load_split_data("test")
    val_examples = load_split_data("val")
    vocabs, all_ids = load_vocab()

    mrrs = []
    hits1 = []
    hits3 = []
    hits10 = []
    for trial in range(5):
        # mrr, hit1, hit3, hit10 = test_lookup_table(test_examples, train_examples, vocabs, all_ids)
        # mrr, hit1, hit3, hit10 = test_random(test_examples, vocabs, all_ids)
        # mrr, hit1, hit3, hit10 = test_mode(test_examples, train_examples, vocabs, all_ids)
        # mrr, hit1, hit3, hit10 =  test_frequency(test_examples, train_examples, vocabs, all_ids)
        mrr, hit1, hit3, hit10 = test_lookup_table_frequency(
            val_examples, train_examples, vocabs, all_ids, data_augmentation=data_augmentation
        )
        mrrs.append(mrr)
        hits1.append(hit1)
        hits3.append(hit3)
        hits10.append(hit10)
    print("mrr:", np.mean(mrrs), np.std(mrrs))
    print("hit@1:", np.mean(hits1), np.std(hits1))
    print("hit@3:", np.mean(hits3), np.std(hits3))
    print("hit@10:", np.mean(hits10), np.std(hits10))
