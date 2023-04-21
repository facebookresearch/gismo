# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List


class FakeRawDataset:
    """
    Utility to create the differents parts of the raw Recipe1M dataset
    """

    def __init__(self):
        self.dets = []
        self.layer1 = []
        self.layer2 = []

    def add(
        self,
        id: str,
        title: str,
        partition: str,
        ingredients: List[str],
        instructions: List[str],
        images: List[str],
    ):
        self.dets.append(self.create_dets(id, ingredients))
        self.layer1.append(
            self.create_layer1(id, title, partition, ingredients, instructions)
        )
        self.layer2.append(self.create_layer2(id, images))

    @staticmethod
    def create_dets(id: str, ingredients: List[str]):
        return {
            "id": id,
            "valid": [True] * len(ingredients),
            "ingredients": [{"text": ingr} for ingr in ingredients],
        }

    @staticmethod
    def create_layer1(
        id: str,
        title: str,
        partition: str,
        ingredients: List[str],
        instructions: List[str],
    ):
        return {
            "id": id,
            "title": title,
            "partition": partition,
            "url": "http://" + id + ".html",
            "ingredients": [{"text": ingr} for ingr in ingredients],
            "instructions": [{"text": instr} for instr in instructions],
        }

    @staticmethod
    def create_layer2(id: str, images: List[str]):
        return {
            "id": id,
            "images": [{"id": img, "url": "http://" + img} for img in images],
        }
