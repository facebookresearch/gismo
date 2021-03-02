# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
    SetPredictionType,
)
from inv_cooking.models.modules.rnn_decoder import DecoderRNN
from inv_cooking.models.modules.transformer_decoder import DecoderTransformer

from .modules.permutation_invariant_criterion import (
    BiPartiteAssignmentCriterion,
    PooledBinaryCrossEntropy,
    ProbaChamferDistance,
)
from .predictor import IngredientsPredictor
from .utils import mask_from_eos


class AutoRegressiveIngredientsPredictor(IngredientsPredictor):
    """
    Implementation of an ingredient predictor based on an auto-regressive architecture
    """

    @classmethod
    def create_tf_from_config(
        cls,
        config: IngredientPredictorTransformerConfig,
        max_num_ingredients: int,
        vocab_size: int,
        eos_value: int,
    ) -> "AutoRegressiveIngredientsPredictor":
        max_num_ingredients += 1  # required for EOS token
        print(
            "Building Transformer decoder {}. Embed size {} / Dropout {} / Max. Num. Labels {} / "
            "Num. Attention Heads {} / Num. Layers {} / Activation {}.".format(
                config.model.name,
                config.embed_size,
                config.dropout,
                max_num_ingredients,
                config.n_att,
                config.layers,
                config.activation,
            ),
            flush=True,
        )
        decoder = DecoderTransformer(
            config.embed_size,
            vocab_size,
            dropout=config.dropout,
            seq_length=max_num_ingredients,
            attention_nheads=config.n_att,
            pos_embeddings=False,
            num_layers=config.layers,
            learned=False,
            activation=config.activation,
        )
        return cls.from_decoder(
            config, decoder, max_num_ingredients, vocab_size, eos_value
        )

    @classmethod
    def create_lstm_from_config(
        cls,
        config: IngredientPredictorLSTMConfig,
        max_num_ingredients: int,
        vocab_size: int,
        eos_value: int,
    ) -> "AutoRegressiveIngredientsPredictor":
        max_num_ingredients += 1  # required for EOS token
        print(
            "Building LSTM decoder {}. Embed size {} / Dropout {} / Max. Num. Labels {}. ".format(
                config.model.name,
                config.embed_size,
                config.dropout,
                max_num_ingredients,
            ),
            flush=True,
        )
        decoder = DecoderRNN(
            config.embed_size,
            config.embed_size,
            vocab_size,
            dropout=config.dropout,
            seq_length=max_num_ingredients,
        )
        return cls.from_decoder(
            config, decoder, max_num_ingredients, vocab_size, eos_value
        )

    @staticmethod
    def from_decoder(
        config: IngredientPredictorLSTMConfig,
        decoder: nn.Module,
        max_num_ingredients: int,
        vocab_size: int,
        eos_value: int,
    ):
        pad_value = vocab_size - 1
        if config.with_set_prediction == SetPredictionType.pooled_bce:
            criterion = PooledBinaryCrossEntropy(
                eos_value=eos_value, pad_value=vocab_size - 1
            )
        elif config.with_set_prediction == SetPredictionType.bipartite:
            criterion = BiPartiteAssignmentCriterion(
                eos_value=eos_value, pad_value=vocab_size - 1
            )
        elif config.with_set_prediction == SetPredictionType.chamfer_l2:
            criterion = ProbaChamferDistance(
                eos_value=eos_value, pad_value=vocab_size - 1
            )
        else:
            criterion = nn.CrossEntropyLoss(ignore_index=pad_value, reduction="mean")

        return AutoRegressiveIngredientsPredictor(
            decoder,
            max_num_ingredients=max_num_ingredients,
            vocab_size=vocab_size,
            criterion=criterion,
            pad_value=pad_value,
            eos_value=eos_value,
            permutation_invariant=config.with_set_prediction != SetPredictionType.none,
        )

    def __init__(
        self,
        decoder: nn.Module,
        max_num_ingredients: int,
        vocab_size: int,
        criterion=None,
        pad_value: int = 0,
        eos_value: int = 0,
        permutation_invariant: bool = True,
    ):
        super().__init__(requires_eos=False)
        self.decoder = decoder
        self.max_num_ingredients = max_num_ingredients
        self.criterion = criterion
        self.permutation_invariant = permutation_invariant
        self.pad_value = pad_value
        self.eos_value = eos_value
        self.vocab_size = vocab_size

    def _forward_impl(
        self,
        img_features: torch.Tensor,
        label_target: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        # use auto-regressive decoder to predict labels (sample function)
        # output label_logits is only used to compute losses in case of self.perminv (no teacher forcing)
        # predictions output is used for all auto-regressive models
        predictions, label_logits = self.decoder.sample(
            img_features, None, first_token_value=0, replacement=False,
        )

        if compute_predictions:
            # mask labels after finding eos
            sample_mask = mask_from_eos(
                predictions, eos_value=self.eos_value, mult_before=False
            )
            predictions[sample_mask == 0] = self.pad_value

        if compute_losses:
            if self.permutation_invariant:
                # autoregressive mode for decoder when training with permutation invariant objective
                losses = self.criterion(label_logits, label_target)

            else:
                # other autoregressive models
                losses: Dict[str, torch.Tensor] = {}

                # add dummy first word to sequence and remove last
                first_word = torch.zeros(len(label_target)).type_as(label_target)
                shift_target = torch.cat([first_word.unsqueeze(-1), label_target], -1)[
                    :, :-1
                ]

                # we need to recompute logits using teacher forcing (forward pass)
                label_logits, _ = self.decoder(img_features, None, shift_target)
                label_logits_v = label_logits.view(
                    label_logits.size(0) * label_logits.size(1), -1
                )

                # compute label loss
                label_target_v = label_target.view(-1)
                loss = self.criterion(label_logits_v, label_target_v)
                losses["label_loss"] = loss

        return losses, predictions
