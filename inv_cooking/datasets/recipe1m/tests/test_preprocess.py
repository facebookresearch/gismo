import math
from collections import Counter

import pytest
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

    # TODO - this is a bug: the tomato is conflicted with tomato_sauce and mushroom_cap with mushroom_caps
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

    assert len(vocab_ingrs.word2idx.keys()) == 9  # TODO - there should only be 5 (bug with base words seen above)
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

    # Check that the second recipe being missing makes the whole vocabulary collapse and no sample is retained
    for c in "abde":
        assert c not in vocab_ingrs.word2idx
    assert "c" in vocab_ingrs.word2idx  # TODO - this is a bug: correct this
    assert len(dataset["train"]) == 0
    assert len(dataset["test"]) == 0
    assert len(dataset["val"]) == 0


@pytest.mark.skip
def test_bad_case():
    dets = [
        {
            "id": "000018c8a5",
            "valid": [
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                False,
                False,
                True,
                True,
                True,
                False,
            ],
            "ingredients": [
                {"text": "penne"},
                {"text": "cheese sauce"},
                {"text": "cheddar cheese"},
                {"text": "gruyere cheese"},
                {"text": "dried chipotle powder"},
                {"text": "unsalted butter"},
                {"text": "all - purpose flour"},
                {"text": "milk"},
                {
                    "text": "14 ounces semihard cheese (page 23), grated (about 3 1/2 cups)"
                },
                {"text": "2 ounces semisoft cheese (page 23), grated (1/2 cup)"},
                {"text": "kosher salt"},
                {"text": "dried chipotle powder"},
                {"text": "garlic powder"},
                {"text": "(makes about 4 cups)"},
            ],
        }
    ]

    layer1 = [
        {
            "id": "000018c8a5",
            "title": "Worlds Best Mac and Cheese",
            "partition": "train",
            "url": "http://www.epicurious.com/recipes/food/views/-world-s-best-mac-and-cheese-387747",
            "ingredients": [
                {"text": "6 ounces penne"},
                {"text": "2 cups Beechers Flagship Cheese Sauce (recipe follows)"},
                {"text": "1 ounce Cheddar, grated (1/4 cup)"},
                {"text": "1 ounce Gruyere cheese, grated (1/4 cup)"},
                {"text": "1/4 to 1/2 teaspoon chipotle chili powder (see Note)"},
                {"text": "1/4 cup (1/2 stick) unsalted butter"},
                {"text": "1/3 cup all-purpose flour"},
                {"text": "3 cups milk"},
                {
                    "text": "14 ounces semihard cheese (page 23), grated (about 3 1/2 cups)"
                },
                {"text": "2 ounces semisoft cheese (page 23), grated (1/2 cup)"},
                {"text": "1/2 teaspoon kosher salt"},
                {"text": "1/4 to 1/2 teaspoon chipotle chili powder"},
                {"text": "1/8 teaspoon garlic powder"},
                {"text": "(makes about 4 cups)"},
            ],
            "instructions": [
                {
                    "text": "Preheat the oven to 350 F. Butter or oil an 8-inch baking dish."
                },
                {"text": "Cook the penne 2 minutes less than package directions."},
                {"text": "(It will finish cooking in the oven.)"},
                {"text": "Rinse the pasta in cold water and set aside."},
                {
                    "text": "Combine the cooked pasta and the sauce in a medium bowl and mix carefully but thoroughly."
                },
                {"text": "Scrape the pasta into the prepared baking dish."},
                {
                    "text": "Sprinkle the top with the cheeses and then the chili powder."
                },
                {"text": "Bake, uncovered, for 20 minutes."},
                {"text": "Let the mac and cheese sit for 5 minutes before serving."},
                {
                    "text": "Melt the butter in a heavy-bottomed saucepan over medium heat and whisk in the flour."
                },
                {"text": "Continue whisking and cooking for 2 minutes."},
                {"text": "Slowly add the milk, whisking constantly."},
                {
                    "text": "Cook until the sauce thickens, about 10 minutes, stirring frequently."
                },
                {"text": "Remove from the heat."},
                {"text": "Add the cheeses, salt, chili powder, and garlic powder."},
                {
                    "text": "Stir until the cheese is melted and all ingredients are incorporated, about 3 minutes."
                },
                {"text": "Use immediately, or refrigerate for up to 3 days."},
                {
                    "text": "This sauce reheats nicely on the stove in a saucepan over low heat."
                },
                {"text": "Stir frequently so the sauce doesnt scorch."},
                {
                    "text": "This recipe can be assembled before baking and frozen for up to 3 monthsjust be sure to use a freezer-to-oven pan and increase the baking time to 50 minutes."
                },
                {
                    "text": "One-half teaspoon of chipotle chili powder makes a spicy mac, so make sure your family and friends can handle it!"
                },
                {
                    "text": "The proportion of pasta to cheese sauce is crucial to the success of the dish."
                },
                {
                    "text": "It will look like a lot of sauce for the pasta, but some of the liquid will be absorbed."
                },
            ],
        }
    ]

    layer2 = [
        {
            "id": "000018c8a5",
            "images": [
                {
                    "id": "3e233001e2.jpg",
                    "url": "http://img.sndimg.com/food/image/upload/w_512,h_512,c_fit,fl_progressive,q_95/v1/img/recipes/47/91/49/picaYYmb9.jpg",
                },
                {
                    "id": "7f749987f9.jpg",
                    "url": "http://img.sndimg.com/food/image/upload/w_512,h_512,c_fit,fl_progressive,q_95/v1/img/recipes/47/91/49/picpy37SW.jpg",
                },
                {
                    "id": "aaf6b2dcd3.jpg",
                    "url": "http://img.sndimg.com/food/image/upload/w_512,h_512,c_fit,fl_progressive,q_95/v1/img/recipes/47/91/49/picX9CNE2.jpg",
                },
            ],
        }
    ]

    config = DictConfig(
        {
            "threshold_ingrs": 1,  # minimum ingr count threshold
            "threshold_words": 1,  # minimum word count threshold
            "maxnuminstrs": 100,  # max number of instructions (sentences)
            "maxnumingrs": 100,  # max number of ingredients
            "minnuminstrs": 1,  # min number of instructions (sentences)
            "minnumingrs": 1,  # min number of ingredients
            "minnumwords": 1,  # minimum number of characters in recipe)
        }
    )

    vocab_ingrs, vocab_toks, dataset = build_vocab_recipe1m(
        dets, layer1, layer2, config
    )
    print(vocab_ingrs.word2idx)
    print(vocab_toks.word2idx)
    print(dataset)

    """
    dets, layer1, layer2 = load_unprocessed_dataset("/checkpoint/qduval/recipe1m/")
    print(dets[0])
    print(layer1[0])
    print(layer2[0])
    """
