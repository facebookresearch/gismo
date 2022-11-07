# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum

from omegaconf import MISSING


class EncoderAttentionType(Enum):
    concat = 0
    seq_img_first = 1
    seq_ingr_first = 2
    concat_tf = 3


@dataclass
class RecipeGeneratorConfig:
    dropout: float = MISSING
    embed_size: int = MISSING
    n_att_heads: int = MISSING
    tf_dec_layers: int = MISSING
    activation: str = MISSING
    encoder_attn: EncoderAttentionType = MISSING
    tf_enc_layers: int = MISSING
