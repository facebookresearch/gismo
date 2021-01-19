# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import (
    CardinalityPredictionType,
    IngredientPredictorCriterion,
    IngredientPredictorFFConfig,
)
from inv_cooking.models.modules.ff_decoder import FFDecoder
from inv_cooking.models.modules.utils import freeze_fn
from inv_cooking.utils.criterion import SoftIoUCriterion, TargetDistributionCriterion

from .predictor import IngredientsPredictor
from .utils import label2_k_hots, predictions_to_indices


class FeedForwardIngredientsPredictor(IngredientsPredictor):
    """
    Implementation of an ingredient predictor based on a feed-forward architecture
    """

    @staticmethod
    def from_config(
        config: IngredientPredictorFFConfig, max_num_ingredients: int, vocab_size: int
    ) -> "FeedForwardIngredientsPredictor":
        cardinality_pred = config.cardinality_pred
        print(
            "Building feed-forward decoder {}. Embed size {} / Dropout {} / "
            " Max. Num. Ingredients {} / Num. Layers {}".format(
                config.model.name,
                config.embed_size,
                config.dropout,
                max_num_ingredients,
                config.layers,
            ),
            flush=True,
        )
        decoder = FFDecoder(
            embed_size=config.embed_size,
            vocab_size=vocab_size,
            hidden_size=config.embed_size,
            dropout=config.dropout,
            n_layers=config.layers,
        )

        # cardinality loss
        if cardinality_pred == CardinalityPredictionType.categorical:
            print("Using categorical cardinality loss.", flush=True)
            decoder.add_cardinality_prediction(max_num_ingredients)
            cardinality_loss = nn.CrossEntropyLoss(reduction="mean")
        else:
            print("Using no cardinality loss.", flush=True)
            cardinality_loss = None

        # label and eos loss
        label_losses = {
            IngredientPredictorCriterion.bce: nn.BCEWithLogitsLoss(reduction="mean"),
            IngredientPredictorCriterion.iou: SoftIoUCriterion(reduction="mean"),
            IngredientPredictorCriterion.td: TargetDistributionCriterion(reduction="mean"),
        }
        label_loss = label_losses[config.criterion]

        model = FeedForwardIngredientsPredictor(
            decoder,
            max_num_ingredients=max_num_ingredients,
            vocab_size=vocab_size,
            crit=label_loss,
            crit_cardinality=cardinality_loss,
            pad_value=vocab_size - 1,
            crit_type=config.criterion,
            card_type=cardinality_pred,
        )
        
        return model

    def __init__(
        self,
        decoder: nn.Module,
        max_num_ingredients: int,
        vocab_size: int,
        crit=None,
        crit_cardinality=None,
        pad_value: int = 0,
        threshold: float = 0.5,
        crit_type: IngredientPredictorCriterion = IngredientPredictorCriterion.bce,
        card_type: CardinalityPredictionType = CardinalityPredictionType.none,
        eps: float = 1e-8,
    ):
        super().__init__(requires_eos=True)
        self.decoder = decoder
        self.max_num_ingredients = max_num_ingredients
        self.crit = crit
        self.threshold = threshold
        self.pad_value = pad_value
        self.crit_cardinality = crit_cardinality
        self.crit_type = crit_type
        self.card_type = card_type
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
        label_logits, cardinality_logits = self.decoder(img_features)

        if compute_losses:
            target_k_hot = label2_k_hots(label_target, self.pad_value)
            target_k_hot = target_k_hot.type_as(label_logits)
            cardinality_target = target_k_hot.sum(dim=-1).unsqueeze(1)

            # compute labels loss
            losses["label_loss"] = self.crit(label_logits, target_k_hot)

            # compute cardinality loss if needed
            if self.crit_cardinality is not None:
                # subtract 1 from num_target because 1st label corresponds to value 1
                offset = 1
                losses["cardinality_loss"] = self.crit_cardinality(
                    cardinality_logits,
                    (cardinality_target.squeeze() - offset).long(),
                )

        if compute_predictions:
            # consider cardinality
            if self.card_type == CardinalityPredictionType.categorical:
                cardinality = nn.functional.log_softmax(
                    cardinality_logits + self.eps, dim=-1
                )
            else:
                cardinality = None

            # apply non-linearity to label logits
            if self.crit_type == IngredientPredictorCriterion.td:
                label_probs = nn.functional.softmax(label_logits, dim=-1)
            else:
                label_probs = torch.sigmoid(label_logits)

            # get label ids
            predictions = predictions_to_indices(
                label_probs=label_probs,
                max_num_labels=self.max_num_ingredients,
                pad_value=self.pad_value,
                threshold=self.threshold,
                cardinality_prediction=cardinality,
                which_loss=self.crit_type,
            )

        return losses, predictions
