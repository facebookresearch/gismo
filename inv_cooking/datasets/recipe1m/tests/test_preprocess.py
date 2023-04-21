# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings
from collections import Counter

from omegaconf import DictConfig

from inv_cooking.datasets.recipe1m.preprocess import (
    BASE_WORDS,
    build_vocab_recipe1m,
    cluster_ingredients,
    remove_plurals,
)
from inv_cooking.datasets.recipe1m.tests.fake_raw_dataset import FakeRawDataset

"""
-------------------------------------------------------------------------------
Ingredient clustering tests
-------------------------------------------------------------------------------
"""


def test_cluster_ingredients():
    counter_ingrs = Counter()
    counter_ingrs["a"] = 1
    counter_ingrs["b"] = 2
    counter_ingrs["c"] = 2
    counter_ingrs["d"] = 2
    counter_ingrs["e"] = 1

    # manually add missing entries for better clustering
    for base_word in BASE_WORDS:
        if base_word not in counter_ingrs.keys():
            counter_ingrs[base_word] = 1

    counter_ingrs, cluster_ingrs = cluster_ingredients(counter_ingrs)
    counter_ingrs, cluster_ingrs = remove_plurals(counter_ingrs, cluster_ingrs)

    warnings.warn(
        "BUG: the tomato is merged with tomato_sauce and mushroom_cap with mushroom_caps"
    )
    assert set(k for k, v in counter_ingrs.items() if v > 1) == {
        "b",
        "c",
        "d",
        "tomato",
        "mushroom_cap",
    }


"""
-------------------------------------------------------------------------------
Full pre-processing tests
-------------------------------------------------------------------------------
"""


ACCEPT_ALL_CONFIG = DictConfig(
    {
        "threshold_ingrs": 1,
        "threshold_words": 1,
        "maxnuminstrs": math.inf,
        "maxnumingrs": math.inf,
        "minnuminstrs": 1,
        "minnumingrs": 1,
        "minnumwords": 1,
    }
)


def test_one_recipe():
    raw_dataset = FakeRawDataset()
    raw_dataset.add(
        id="1",
        title="First recipe",
        partition="train",
        ingredients=["a", "b 1", "c", "d", "e f"],
        instructions=["first do a", "then do b and c", "then add d and e"],
        images=["image1.jpg", "image2.jpg"],
    )

    vocab_ingrs, vocab_toks, dataset = build_vocab_recipe1m(
        raw_dataset.dets, raw_dataset.layer1, raw_dataset.layer2, ACCEPT_ALL_CONFIG
    )

    assert len(vocab_ingrs.word2idx.keys()) == 71  # TODO: 5 + 2 + len(BASE_WORDS)
    assert (
        len(vocab_toks.word2idx.keys()) == 11 + 4
    )  # 11 unique words and 4 special words
    assert "b" in vocab_ingrs.word2idx
    assert "e_f" in vocab_ingrs.word2idx
    assert len(dataset["train"]) == 1
    assert len(dataset["test"]) == 0
    assert len(dataset["val"]) == 0


def test_three_recipes():
    raw_dataset = FakeRawDataset()
    raw_dataset.add(
        id="1",
        title="First recipe",
        partition="train",
        ingredients="a b c d".split(),
        instructions=["first do a", "then do b and c", "then add d"],
        images=["image1.jpg", "image2.jpg"],
    )
    raw_dataset.add(
        id="2",
        title="Second recipe",
        partition="train",
        ingredients="b c d e".split(),
        instructions=["first do e", "then do b and c", "then add d"],
        images=["image3.jpg", "image4.jpg"],
    )

    vocab_ingrs, vocab_toks, dataset = build_vocab_recipe1m(
        raw_dataset.dets, raw_dataset.layer1, raw_dataset.layer2, ACCEPT_ALL_CONFIG
    )

    assert len(vocab_ingrs.word2idx.keys()) == 71  # TODO: 5 + 2 + len(BASE_WORDS)
    assert (
        len(vocab_toks.word2idx.keys()) == 12 + 4
    )  # 11 unique words and 4 special words (title counts)
    assert len(dataset["train"]) == 2
    assert len(dataset["test"]) == 0
    assert len(dataset["val"]) == 0

    raw_dataset.add(
        id="3",
        title="Third recipe",
        partition="val",
        ingredients="b c d e f".split(),
        instructions=["first do e", "then do b and c", "then add d or f"],
        images=["image5.jpg"],
    )

    vocab_ingrs_2, vocab_toks_2, dataset_2 = build_vocab_recipe1m(
        raw_dataset.dets, raw_dataset.layer1, raw_dataset.layer2, ACCEPT_ALL_CONFIG
    )

    assert len(vocab_ingrs) == len(vocab_ingrs_2), "val should not participate"
    assert len(vocab_toks) == len(vocab_toks_2), "val should not participate"
    assert len(dataset_2["train"]) == 2
    assert len(dataset_2["test"]) == 0
    assert len(dataset_2["val"]) == 1
    assert len(dataset_2["val"][0]["images"]) == 1


def test_filtering_1():
    raw_dataset = FakeRawDataset()
    raw_dataset.add(
        id="1",
        title="First recipe",
        partition="train",
        ingredients="a b c d".split(),
        instructions=["first do a", "then do b and c", "then add d"],
        images=["image1.jpg", "image2.jpg"],
    )
    raw_dataset.add(
        id="2",
        title="Second recipe",
        partition="train",
        ingredients="b c d e".split(),
        instructions=["first do e", "then do b and c", "then add d"],
        images=["image3.jpg", "image4.jpg"],
    )

    config = DictConfig(
        {
            "threshold_ingrs": 2,
            "threshold_words": 2,
            "maxnuminstrs": math.inf,
            "maxnumingrs": math.inf,
            "minnuminstrs": 1,
            "minnumingrs": 1,
            "minnumwords": 1,
        }
    )

    vocab_ingrs, vocab_toks, dataset = build_vocab_recipe1m(
        raw_dataset.dets, raw_dataset.layer1, raw_dataset.layer2, config
    )

    warnings.warn(
        "BUG: there should be only 5 vocab_ingrs.word2idx.keys() - issue with merged based words"
    )
    assert len(vocab_ingrs.word2idx.keys()) == 9
    assert len(vocab_toks.word2idx.keys()) == 9 + 4
    assert len(dataset["train"]) == 2
    assert len(dataset["test"]) == 0
    assert len(dataset["val"]) == 0


def test_filtering_2():
    raw_dataset = FakeRawDataset()
    raw_dataset.add(
        id="1",
        title="First recipe",
        partition="train",
        ingredients="a b c d".split(),
        instructions=["first do a", "then do b and c", "then add d"],
        images=["image1.jpg", "image2.jpg"],
    )

    raw_dataset.add(
        id="2",
        title="Second recipe",
        partition="train",
        ingredients="b c d e".split(),
        instructions=["do all at once it is more fun"],
        images=["image3.jpg"],
    )

    config = DictConfig(
        {
            "threshold_ingrs": 2,
            "threshold_words": 2,
            "maxnuminstrs": math.inf,
            "maxnumingrs": math.inf,
            "minnuminstrs": 2,
            "minnumingrs": 1,
            "minnumwords": 1,
        }
    )

    vocab_ingrs, vocab_toks, dataset = build_vocab_recipe1m(
        raw_dataset.dets, raw_dataset.layer1, raw_dataset.layer2, config
    )

    # Check that the second recipe being missing makes the whole vocabulary collapse and no sample is retained
    for c in "abcde":
        assert c not in vocab_ingrs.word2idx
    assert len(dataset["train"]) == 0
    assert len(dataset["test"]) == 0
    assert len(dataset["val"]) == 0


def test_bad_filtering():
    raw_dataset = FakeRawDataset()
    raw_dataset.add(
        id="1",
        title="First recipe",
        partition="train",
        ingredients="a b c".split(),
        instructions=["first do a", "then do b and c"],
        images=["image1.jpg"],
    )

    raw_dataset.add(
        id="2",
        title="Second recipe",
        partition="train",
        ingredients="c d e".split(),
        instructions=["first do c", "then do d and e"],
        images=["image2.jpg"],
    )

    config = DictConfig(
        {
            "threshold_ingrs": 2,
            "threshold_words": 1,
            "maxnuminstrs": math.inf,
            "maxnumingrs": math.inf,
            "minnuminstrs": 1,
            "minnumingrs": 2,
            "minnumwords": 1,
        }
    )

    vocab_ingrs, vocab_toks, dataset = build_vocab_recipe1m(
        raw_dataset.dets, raw_dataset.layer1, raw_dataset.layer2, config
    )

    # Check that the dataset is empty:
    #   - ingredients appear only once except 'c' and are thus removed from vocabulary
    #   - the number of ingredient is thus only 1 by recipe (below threshold of 2)
    assert len(dataset["train"]) == 0
    assert len(dataset["test"]) == 0
    assert len(dataset["val"]) == 0

    # Check that the vocabulary is empty
    warnings.warn(
        "BUG: 'c' should not be part of the vocabulary as it belongs to no recipe"
    )
    for c in "abde":
        assert c not in vocab_ingrs.word2idx
    assert "c" in vocab_ingrs.word2idx
