from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.utils.metrics.ingredient_iou import TrieIngredientExtractor


def test_ingredient_extractor():
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

    recipe = ' mincemeat scrambled eggs with herbs and pepper \n add some low fat milk to the sauce\n in a small bowl , whisk together eggs , cream , salt and herbs .\n heat butter in a large nonstick skillet over medium heat .\n add eggs and cook , stirring constantly , until eggs are set but still moist , about 3 minutes .\n remove from heat and stir in herbs .\n serve immediately .\n'
    recipe_words = recipe.split(" ")

    extractor = TrieIngredientExtractor(vocab)

    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=set())
    assert ingr == ['cream', 'egg', 'herb', 'low_fat_milk', 'mincemeat', 'pepper', 'salt']

    preferred_ingredients = {vocab.word2idx["cayenne_pepper"]}
    _, ingr = extractor.viz_ingredients(recipe_words, preferred_ingredients=preferred_ingredients)
    assert ingr == ['cayenne_pepper', 'cream', 'egg', 'herb', 'low_fat_milk', 'mincemeat', 'salt']
