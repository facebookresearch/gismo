# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
)
from inv_cooking.models.modules.rnn_decoder import DecoderRNN
from inv_cooking.models.modules.transformer_decoder import DecoderTransformer
from inv_cooking.models.modules.utils import freeze_fn
from .predictor import IngredientsPredictor
from .utils import mask_from_eos, label2_k_hots


class AutoRegressiveIngredientsPredictor(IngredientsPredictor):
    """
    Implementation of an ingredient predictor based on an auto-regressive architecture
    """

    @classmethod
    def create_tf_from_config(
            cls,
            config: IngredientPredictorTransformerConfig,
            max_num_labels: int,
            vocab_size: int,
            eos_value: int,
    ) -> "AutoRegressiveIngredientsPredictor":
        max_num_labels += 1  # required for EOS token
        print(
            "Building Transformer decoder {}. Embed size {} / Dropout {} / Max. Num. Labels {} / "
            "Num. Attention Heads {} / Num. Layers {}.".format(
                config.model,
                config.embed_size,
                config.dropout,
                max_num_labels,
                config.n_att,
                config.layers,
            ),
            flush=True,
        )
        decoder = DecoderTransformer(
            config.embed_size,
            vocab_size,
            dropout=config.dropout,
            seq_length=max_num_labels,
            attention_nheads=config.n_att,
            pos_embeddings=False,
            num_layers=config.layers,
            learned=False,
            normalize_before=True,
        )
        return cls.from_decoder(config, decoder, max_num_labels, vocab_size, eos_value)

    @classmethod
    def create_lstm_from_config(
            cls,
            config: IngredientPredictorLSTMConfig,
            max_num_labels: int,
            vocab_size: int,
            eos_value: int,
    ) -> "AutoRegressiveIngredientsPredictor":
        max_num_labels += 1  # required for EOS token
        print(
            "Building LSTM decoder {}. Embed size {} / Dropout {} / Max. Num. Labels {}. ".format(
                config.model, config.embed_size, config.dropout, max_num_labels
            ),
            flush=True,
        )
        decoder = DecoderRNN(
            config.embed_size,
            config.embed_size,
            vocab_size,
            dropout=config.dropout,
            seq_length=max_num_labels,
        )
        return cls.from_decoder(config, decoder, max_num_labels, vocab_size, eos_value)

    @staticmethod
    def from_decoder(
            config: IngredientPredictorLSTMConfig,
            decoder: nn.Module,
            max_num_labels: int,
            vocab_size: int,
            eos_value: int):
        pad_value = vocab_size - 1
        if config.with_set_prediction:
            label_loss = nn.BCELoss(reduction="mean")
            eos_loss = nn.BCELoss(reduction="none")
        else:
            label_loss = nn.CrossEntropyLoss(ignore_index=pad_value, reduction="mean")
            eos_loss = None

        model = AutoRegressiveIngredientsPredictor(
            decoder,
            max_num_labels,
            vocab_size,
            crit=label_loss,
            crit_eos=eos_loss,
            pad_value=pad_value,
            eos_value=eos_value,
            perminv=config.with_set_prediction,
        )

        if config.freeze:
            freeze_fn(model)
        return model

    def __init__(
            self,
            decoder: nn.Module,
            max_num_labels: int,
            vocab_size: int,
            crit=None,
            crit_eos=None,
            pad_value: int = 0,
            eos_value: int = 0,
            perminv: bool = True,
            eps: float = 1e-8,
    ):
        super().__init__(remove_eos=False)
        self.decoder = decoder
        self.maxnumlabels = max_num_labels
        self.crit = crit
        self.perminv = perminv
        self.pad_value = pad_value
        self.eos_value = eos_value
        self.crit_eos = crit_eos
        self.eps = eps
        self.vocab_size = vocab_size

    def _forward_impl(
            self,
            img_features: torch.Tensor,
            label_target: Optional[torch.Tensor] = None,
            compute_losses: bool = False,
            compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        losses: Dict[str, torch.Tensor] = {}
        predictions: Optional[torch.Tensor] = None

        # use auto-regressive decoder to predict labels (sample function)
        # output label_logits is only used to compute losses in case of self.perminv (no teacher forcing)
        # predictions output is used for all auto-regressive models
        predictions, label_logits = self.decoder.sample(
            img_features, None, first_token_value=0, replacement=False,
        )

        if compute_predictions:
            # mask labels after finding eos (cardinality)
            sample_mask = mask_from_eos(
                predictions, eos_value=self.eos_value, mult_before=False
            )
            predictions[sample_mask == 0] = self.pad_value

        if compute_losses:
            # add dummy first word to sequence and remove last
            first_word = torch.zeros(len(label_target)).type_as(label_target)
            shift_target = torch.cat([first_word.unsqueeze(-1), label_target], -1)[
                           :, :-1
                           ]

            # autoregressive mode for decoder when training with permutation invariant objective
            if self.perminv:

                # apply softmax non-linearity before pooling across timesteps
                label_probs = nn.functional.softmax(label_logits, dim=-1)

                # Find index for eos label
                # eos probability is the one assigned to the position self.eos_value of the softmax
                # this is used with bce loss only
                eos = label_probs[:, :, self.eos_value]

                # all zeros except position where eos is in the gt
                eos_pos = label_target == self.eos_value

                # 1s for gt label positions, 0s starting from eos position in the gt
                eos_head = (label_target != self.pad_value) & (label_target != self.eos_value)

                # 0s for gt label positions, 1s starting from eos position in the gt
                eos_target = ~eos_head

                # select transformer steps to pool (steps corresponding to set elements, i.e. labels)
                label_probs = label_probs * eos_head.float().unsqueeze(-1)

                # pool
                label_probs, _ = torch.max(label_probs, dim=1)

                # compute label loss
                target_k_hot = label2_k_hots(
                    label_target, self.pad_value, remove_eos=True
                )
                loss = self.crit(label_probs[:, 1:], target_k_hot.float())
                losses["label_loss"] = loss

                # compute eos loss
                eos_loss = self.crit_eos(eos, eos_target.float())

                # eos loss is computed for all timesteps <= eos in gt and
                # equally penalizes the head (all 0s) and the true eos position (1)
                losses["eos_loss"] = (
                        0.5
                        * (eos_loss * eos_pos.float()).sum(1)
                        / (eos_pos.float().sum(1) + self.eps)
                        + 0.5
                        * (eos_loss * eos_head.float()).sum(1)
                        / (eos_head.float().sum(1) + self.eps)
                ).mean()

            else:
                # other autoregressive models
                # we need to recompute logits using teacher forcing (forward pass)
                label_logits, _ = self.decoder(img_features, None, shift_target)
                label_logits_v = label_logits.view(
                    label_logits.size(0) * label_logits.size(1), -1
                )

                # compute label loss
                label_target_v = label_target.view(-1)
                loss = self.crit(label_logits_v, label_target_v)
                losses["label_loss"] = loss

        return losses, predictions
