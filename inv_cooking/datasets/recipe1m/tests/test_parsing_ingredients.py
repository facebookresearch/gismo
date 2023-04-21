# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from inv_cooking.datasets.recipe1m.parsing import IngredientParser


def test_parsing_instructions():
    parser = IngredientParser(
        replace_dict={
            "and": ["&", "'n"],
        }
    )

    out = parser.parse_entry(
        {
            "valid": [True, False, True],
            "ingredients": [
                {"text": "egg'n cheese"},
                {"text": "invalid"},
                {"text": "1. once milk"},
            ],
        },
        clean_digits=True,
    )
    assert ["eggand_cheese", "._once_milk"] == list(out)
