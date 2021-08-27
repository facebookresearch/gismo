"""
Was to generate cleaned_yummly_ingredients.json based on ingredients_yummly.json
"""
import json
from pathlib import Path

from normalisation.helpers.recipe_normalizer import RecipeNormalizer

if __name__ == "__main__":
    # Important!: We manually added some ingredients, so if this is rerun, those will be deleted
    yummly_ingredients_path = Path("data/ingredients_yummly.json")
    cleaned_yummly_ingredients_path = Path("data/cleaned_yummly_ingredients.json")
    ingredient_normalizer = RecipeNormalizer(lemmatization_types=["NOUN"])

    with yummly_ingredients_path.open() as f:
        ingredients_yummly = json.load(f)

    print(f"Yummly Ingredients Size: {len(list(set(ingredients_yummly)))}")
    # singulatisation/lemmatization of nouns
    cleaned_ingredients = list(
        set(ingredient_normalizer.normalize_ingredients(ingredients_yummly))
    )
    # Currently only deletes if number is single digit, lets see if this causes any problems

    print(f"Cleaned Ingredients Size: {len(cleaned_ingredients)}")
    # delete all with more than 3 words
    cleaned_yummly_ingredients = sorted(
        list(
            {
                elem.strip()
                for elem in cleaned_ingredients
                if len(elem.split()) <= 3 and len(elem) > 1
            }
        )
    )
    print(f"Final Cleaned Ingredient size: {len(cleaned_yummly_ingredients)}")

    with cleaned_yummly_ingredients_path.open("w") as f:
        json.dump(cleaned_yummly_ingredients, f)
