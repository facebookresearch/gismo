# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from inv_cooking.models.modules.ingredient_embeddings import IngredientEmbeddings


def test_ingredient_encoder():
    encoder = IngredientEmbeddings(embed_size=2048, voc_size=10)
    x = torch.LongTensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]])
    y = encoder(x)
    assert y.shape == torch.Size([3, 2048, 4])


def test_ingredient_encoder_one_hot():
    encoder = IngredientEmbeddings(embed_size=2048, voc_size=3)
    x = torch.FloatTensor([[[0, 0, 1], [0, 1, 0], [1, 0, 0]]])
    y = encoder(x, onehot_flag=True)
    assert y.shape == torch.Size([1, 2048, 3])
