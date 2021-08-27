def has_number(token):
    # check if string has any numbers
    return any(char.isdigit() for char in token.text)


words_to_remove = [
    "/",
    "-",
    "ounce",
    "cup",
    "teaspoon",
    "tbsp",
    "tsp",
    "tablespoon",
    "sm",
    "c",
    "cube",
    "tbsp.",
    "sm.",
    "c.",
    "oz",
]
