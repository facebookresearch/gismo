from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.utils.metrics.ingredient_iou import TrieIngredientExtractor


def stub_vocabulary():
    vocab = Vocabulary()
    vocab.add_word_group(["low_fat_milk", "1%_milk"])
    vocab.add_word_group(["milk"])
    vocab.add_word_group(["black_pepper", "ground_black_pepper"])
    vocab.add_word_group(["cayenne_pepper"])
    vocab.add_word_group(["pepper"])
    vocab.add_word_group(["egg", "eggs", "duck_egg"])
    vocab.add_word_group(["salt"])
    vocab.add_word_group(["cream"])
    vocab.add_word_group(["herb", "herbs"])
    vocab.add_word_group(["mincemeat"])
    return vocab


def recipe_to_words(text: str):
    return [word for line in text.splitlines() for word in line.split(" ")]


def test_ingredient_extract_find_at_beginning_and_end():
    vocab = stub_vocabulary()
    recipe_words = recipe_to_words("mincemeat bar to mix with one egg or multiple eggs with the cream")
    extractor = TrieIngredientExtractor(vocab)
    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=set())
    assert ingr == ['cream', 'egg', 'mincemeat']


def test_ingredient_extract_find_best_match():
    vocab = stub_vocabulary()
    recipe_words = recipe_to_words("mix everything together with milk and eggs")
    extractor = TrieIngredientExtractor(vocab)
    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=set())
    assert ingr == ['egg', 'milk']
    preferred_ingredients = {vocab.word2idx["low_fat_milk"]}
    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=preferred_ingredients)
    assert ingr == ['egg', 'low_fat_milk']


def test_ingredient_extractor_longer_recipe():
    vocab = stub_vocabulary()

    recipe_words = recipe_to_words("""
    mincemeat scrambled eggs with herbs and pepper \n
    add some low fat milk to the sauce\n
    in a small bowl , whisk together eggs , cream , salt and herbs .\n
    heat butter in a large nonstick skillet over medium heat .\n
    add eggs and cook , stirring constantly , until eggs are set but still moist , about 3 minutes .\n
    remove from heat and stir in herbs .\n
    serve immediately .\n
    """)

    extractor = TrieIngredientExtractor(vocab)

    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=set())
    assert ingr == ['cream', 'egg', 'herb', 'low_fat_milk', 'mincemeat', 'pepper', 'salt']

    preferred_ingredients = {vocab.word2idx["cayenne_pepper"]}
    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=preferred_ingredients)
    assert ingr == ['cayenne_pepper', 'cream', 'egg', 'herb', 'low_fat_milk', 'mincemeat', 'salt']


def test_ingredient_extractor_problematic_cases():
    vocab = Vocabulary()
    vocab.add_word_group(["soup_mix"])
    vocab.add_word_group(["creamed_spinach", "spinach_dip"])
    vocab.add_word_group(["frozen_spinach"])

    extractor = TrieIngredientExtractor(vocab)
    recipe_words = "spinach dip : mix all ingredients together".split(" ")

    preferred_ingredients = {vocab.word2idx["frozen_spinach"]}
    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=preferred_ingredients)
    assert ingr == ["creamed_spinach"]

    preferred_ingredients = {vocab.word2idx["frozen_spinach"], vocab.word2idx["soup_mix"]}
    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=preferred_ingredients)
    assert ingr == ["creamed_spinach", "soup_mix"]
