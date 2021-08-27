## Summary
Scripts to prepare the cleaned ingredients we use and the normalized recipe instructions.

## How it Works 
- Numbers, specific words like measurement units, and everything enclosed in brackets is deleted from the ingredients 
 - Ingredients are all lemmatized and if they contain several words joined with underscore
 - Lemmatisation and joining also is applied to all nouns in the recipe instructions


## How to Run

#### Clean ingredients
These steps are not necessary as we already provide their outputs.
Run
    
    python -m normalisation.generate_final_clean_ingredients
    
A list of the clean ingredients is saved to data/cleaned_yummly_ingredients.json

#### Normalize instructions
Run

    python -m normalisation.normalize_recipe_instructions
    
The recipes with modified instruction are saved to data/cleaned_recipe1m.json