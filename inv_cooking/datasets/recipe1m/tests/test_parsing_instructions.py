# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from inv_cooking.datasets.recipe1m.parsing import InstructionParser
from inv_cooking.datasets.recipe1m.tests.fake_raw_dataset import FakeRawDataset


def test_parsing_instructions():
    raw_dataset = FakeRawDataset()
    raw_dataset.add(
        id="1",
        title="First recipe",
        partition="train",
        ingredients="a b c".split(),
        instructions=["first do a", "then do b & c ", "3. finished!"],
        images=["image1.jpg"],
    )

    parser = InstructionParser(
        replace_dict={
            "and": ["&", "'n"],
        }
    )

    total_len, instructions = parser.parse_entry(raw_dataset.layer1[0])
    assert instructions == ["first do a", "then do b and c"]
    assert total_len == sum(len(instr) for instr in instructions)
