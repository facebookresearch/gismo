# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .ff_ingr_decoder import FFIngredientDecoder


def test_ff_ingr_decoder():
    ff_ingr_decoder = FFIngredientDecoder(
        embed_size=1024, vocab_size=123, hidden_size=2048, dropout=0.1, n_layers=3
    )
    ff_ingr_decoder.add_cardinality_prediction(max_num_ingredients=20)
    x = torch.randn(size=(2, 1024, 4))
    logits, cardinality = ff_ingr_decoder(x)
    assert logits.shape == torch.Size([2, 122])  # Removed one of the EOS
    assert cardinality.shape == torch.Size([2, 20])  # Not able to predict 0
