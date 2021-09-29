import pickle
import random


# remove the possible duplicates in the dataset
def remove_duplicates(set_):
    union = {}
    for example in set_:
        tuple_ = (example["id"], example["text"])
        union[tuple_] = True
    return list(union.keys())


# find all occurances of character ch in string s
def find_char(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


# find ingredients in the comment text previously marked with $ and £
def find_ingredients(text):
    try:
        s1, s2 = find_char(text, "$")[:2]
        t1, t2 = find_char(text, "£")[:2]
        return text[s1 + 2 : s2 - 1], text[t1 + 2 : t2 - 1]
    except Exception:
        return "", ""


# find if the ingredient word appeared in the list of ingredients ings in the recipe
def find_word_ing(word, ings):
    ings = [item for sublist in ings for item in sublist]
    # if word is explicitly in the ings
    if word in ings:
        return True, word
    # if ing is not explicitly in the ings, find out if an specific version of ing is in ings
    # e.g., if word is oil and there is not any oil in ings but there is vegetable_oil in ings
    else:
        super_str = []
        for ing in ings:
            if word in ing:
                super_str.append(ing)
        # if there are multiple matches for word in ing, go with the one ending with ing.
        if len(super_str) > 0:
            for ing in super_str:
                if ing.endswith(word):
                    return True, ing
            return True, random.choice(super_str)
    return False, word


# convert the set of ingredient ids to a set of words
def ids_to_words(list_ing, vocab):
    res = []
    for ing in list_ing:
        try:
            ing_word = vocab.idx2word[vocab.word2idx[ing]]
            if ing_word not in ["<end>", "<pad>"]:
                res.append(ing_word)
        except:
            pass
    return res


# map recipe ids to the ingredients in the recipe
# this is to make sure the substituted ingredient is in the recipe
def read_ingredients_ids(dataloader, vocab_ing):
    id2ingredient = {}
    for data in dataloader:
        curr_ingredients = data["ingredients"]
        id2ingredient[data["id"]] = ids_to_words(curr_ingredients, vocab_ing)
    return id2ingredient


# try to find the longest (more specific) ingredient possible in the text
# e.g., if both oil and vegetable oil are possible ingredients and in the comment we have vegetable oil, go for the vegetable oil
# check both previous word and next word of the ing in the text and try to output the longest possible version
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


# make two ingredients consistent with their types
# if ing1 is vanilla and ing2 is almond_extract, return vanilla_extract
def make_consistent(ing1, ing2, word2idx):
    if "_" in ing2:
        ing2_words = ing2.split("_")
        if ing2_words[-1] not in ing1:
            if ing1 + "_" + ing2_words[-1] in word2idx:
                return ing1 + "_" + ing2_words[-1]
    return ing1


# map comment text to substitutions
def create_substitutions(union, id2ing, vocab_ing, file):
    input_file = open(file, "wb")
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
    pickle.dump(examples, input_file)
    print(len(examples))
    return examples


def create_substitutions_datasets(
    train_dataset,
    val_dataset,
    test_dataset,
    vocab_ings,
    dataset,
    train_dataset_path,
    val_dataset_path,
    test_dataset_path,
):

    train_dataset = remove_duplicates(train_dataset)
    val_dataset = remove_duplicates(val_dataset)
    test_dataset = remove_duplicates(test_dataset)

    test_id2ing = read_ingredients_ids(dataset["test"], vocab_ings)
    val_id2ing = read_ingredients_ids(dataset["val"], vocab_ings)
    train_id2ing = read_ingredients_ids(dataset["train"], vocab_ings)

    train_subs = create_substitutions(
        train_dataset, train_id2ing, vocab_ings, train_dataset_path
    )
    val_subs = create_substitutions(
        val_dataset, val_id2ing, vocab_ings, val_dataset_path
    )
    test_subs = create_substitutions(
        test_dataset, test_id2ing, vocab_ings, test_dataset_path
    )

    return train_subs, val_subs, test_subs
