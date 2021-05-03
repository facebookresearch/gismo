from inv_cooking.datasets.recipe1m.parsing import IngredientParser


def test_parsing_instructions():
    parser = IngredientParser(replace_dict={
        "and": ["&", "'n"],
    })

    out = parser.parse_entry({
        "valid": [True, False, True],
        "ingredients": [{"text": "egg'n cheese"}, {"text": "invalid"}, {"text": "1. once milk"}]
    }, clean_digits=True)
    assert ['eggand_cheese', '._once_milk'] == list(out)
