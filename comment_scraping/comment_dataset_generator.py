import json
import os
import pickle
import random
from pathlib import Path

from utils import get_vocabs, ids_to_words, load_data


def remove_duplicates(set_):
    union = {}
    for example in set_:
        tuple_ = (example["id"], example["text"])
        union[tuple_] = True
    return list(union.keys())


def find_char(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def find_ingredients(text):
    try:
        s1, s2 = find_char(text, "$")[:2]
        t1, t2 = find_char(text, "£")[:2]
        return text[s1 + 2 : s2 - 1], text[t1 + 2 : t2 - 1]
    except Exception:
        return "", ""


def find_word_ing(word, ings):
    ings = [item for sublist in ings for item in sublist]
    if word in ings:
        return True, word
    else:
        super_str = []
        for ing in ings:
            if word in ing:
                super_str.append(ing)
        if len(super_str) > 0:
            for ing in super_str:
                if ing.endswith(word):
                    return True, ing
            return True, random.choice(super_str)
    return False, word


def read_ingredients_ids(dataloader, vocab_ing):
    counter = 0
    id2ingredient = {}
    for batch in dataloader:
        curr_ingredients = batch["ingredients"][0]
        id2ingredient[batch["id"][0]] = ids_to_words(
            curr_ingredients.numpy(), vocab_ing
        )
        counter += 1
    return id2ingredient


def find_longest_ing(text, ing, word2idx):
    words = text.replace("$", " ").split()
    ind_ing = words.index(ing)
    prev_word, next_word = "", ""
    if ind_ing - 1 > 0:
        prev_word = words[ind_ing - 1]
    if ind_ing + 1 < len(words):
        next_word = words[ind_ing + 1]
    if prev_word + "_" + ing + "_" + next_word in word2idx:
        return prev_word + "_" + ing + "_" + next_word
    elif prev_word + "_" + ing + next_word in word2idx:
        return prev_word + "_" + ing + next_word
    elif prev_word + ing + "_" + next_word in word2idx:
        return prev_word + ing + "_" + next_word
    elif prev_word + ing + next_word in word2idx:
        return prev_word + ing + next_word
    elif prev_word + "_" + ing in word2idx:
        return prev_word + "_" + ing
    elif prev_word + ing in word2idx:
        return prev_word + ing
    elif ing + "_" + next_word in word2idx:
        return ing + "_" + next_word
    elif ing + next_word in word2idx:
        return ing + next_word
    return ing


def make_consistent(ing1, ing2, word2idx):
    if "_" in ing2:
        ing2_words = ing2.split("_")
        if ing2_words[-1] not in ing1:
            if ing1 + "_" + ing2_words[-1] in word2idx:
                return ing1 + "_" + ing2_words[-1]
    return ing1


def create_substituions(union, id2ing, vocab_ing, file):
    input_file = open(file, "w")
    examples = []
    counter = 0
    subs = {}
    for review in union:
        if not review[1].endswith("?"):
            text = review[1]
            id_ = review[0]
            ing1, ing2 = find_ingredients(text)
            text_ = text.replace("\u00a3", "$")
            fact1, ing1 = find_word_ing(ing1, id2ing[id_])
            fact2, ing2 = find_word_ing(ing2, id2ing[id_])
            if (
                fact1 ^ fact2
                and ing1 in vocab_ing.word2idx
                and ing2 in vocab_ing.word2idx
            ):
                counter += 1
                ing1_id = vocab_ing.word2idx[ing1]
                ing2_id = vocab_ing.word2idx[ing2]
                if (ing1_id, ing2_id) not in subs:
                    subs[(ing1_id, ing2_id)] = True
                    subs[(ing2_id, ing1_id)] = True
                example = {}
                example["id"] = id_
                example["ingredients"] = id2ing[id_]
                example["text"] = text_
                if fact1:
                    ing2_ = find_longest_ing(text_, ing2, vocab_ing.word2idx)
                    ing2_ = make_consistent(ing2_, ing1, vocab_ing.word2idx)

                    example["subs"] = (ing1, ing2_)

                elif fact2:
                    ing1_ = find_longest_ing(text_, ing1, vocab_ing.word2idx)
                    ing1_ = make_consistent(ing1_, ing2, vocab_ing.word2idx)
                    example["subs"] = (ing2, ing1_)

                if example not in examples:
                    examples.append(example)
    json.dump(examples, input_file)
    return counter, subs, examples


def create_substituions_datasets(
    train_dataset, val_dataset, test_dataset, recipe1m_path, preprocessed_dir
):

    train_id2ing_path = Path(os.path.join(preprocessed_dir, "train_id2ing.pickle"))
    val_id2ing_path = Path(os.path.join(preprocessed_dir, "val_id2ing.pickle"))
    test_id2ing_path = Path(os.path.join(preprocessed_dir, "test_id2ing.pickle"))

    train_dataset_path = Path(os.path.join(recipe1m_path, "train_comments_subs.txt"))
    val_dataset_path = Path(os.path.join(recipe1m_path, "val_comments_subs.txt"))
    test_dataset_path = Path(os.path.join(recipe1m_path, "test_comments_subs.txt"))

    train_dataset = remove_duplicates(train_dataset)
    val_dataset = remove_duplicates(val_dataset)
    test_dataset = remove_duplicates(test_dataset)

    train_dataloader, val_dataloader, test_dataloader, data_module = load_data()
    vocab_ing, _ = get_vocabs(data_module)

    if (
        train_id2ing_path.exists()
        and val_id2ing_path.exists()
        and test_id2ing_path.exists()
    ):
        with train_id2ing_path.open("rb") as handle:
            train_id2ing = pickle.load(handle)
        with val_id2ing_path.open("rb") as handle:
            val_id2ing = pickle.load(handle)
        with test_id2ing_path.open("rb") as handle:
            test_id2ing = pickle.load(handle)
    else:
        test_id2ing = read_ingredients_ids(test_dataloader, vocab_ing)
        val_id2ing = read_ingredients_ids(val_dataloader, vocab_ing)
        train_id2ing = read_ingredients_ids(train_dataloader, vocab_ing)

        with train_id2ing_path.open("wb") as handle:
            pickle.dump(train_id2ing, handle)
        with val_id2ing_path.open("wb") as handle:
            pickle.dump(val_id2ing, handle)
        with test_id2ing_path.open("wb") as handle:
            pickle.dump(test_id2ing, handle)

    create_substituions(train_dataset, train_id2ing, vocab_ing, train_dataset_path)
    create_substituions(val_dataset, val_id2ing, vocab_ing, val_dataset_path)
    create_substituions(test_dataset, test_id2ing, vocab_ing, test_dataset_path)
