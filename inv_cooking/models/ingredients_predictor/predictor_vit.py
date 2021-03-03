from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from inv_cooking.config import IngredientPredictorVITConfig, SetPredictionType

from .modules.permutation_invariant_criterion import (
    BiPartiteAssignmentCriterion,
    ChamferDistanceCriterion,
    ChamferDistanceType,
    PooledBinaryCrossEntropy,
)
from .predictor import IngredientsPredictor
from .utils import mask_from_eos


class VITIngredientsPredictor(IngredientsPredictor):
    """
    A specific ingredient predictor that takes a sequence of image features and predict
    one ingredient for each image feature.

    Loss can either be permutation invariant or not.
    """

    def __init__(
        self,
        config: IngredientPredictorVITConfig,
        max_num_ingredients: int,
        vocab_size: int,
        eos_value: int,
    ):
        super().__init__(requires_eos=True)
        self.max_num_ingredients = max_num_ingredients
        self.eos_value = eos_value
        self.pad_value = vocab_size - 1
        self.decoder = self._create_decoder(config, vocab_size)
        self.permutation_invariant = (
            config.with_set_prediction != SetPredictionType.none
        )
        if config.with_set_prediction == SetPredictionType.bipartite:
            self.criterion = BiPartiteAssignmentCriterion(
                eos_value=self.eos_value, pad_value=self.pad_value
            )
        elif config.with_set_prediction == SetPredictionType.chamfer_l2:
            self.criterion = ChamferDistanceCriterion(
                eos_value=self.eos_value,
                pad_value=self.pad_value,
                distanceType=ChamferDistanceType.l2,
            )
        elif config.with_set_prediction == SetPredictionType.chamfer_ce:
            self.criterion = ChamferDistanceCriterion(
                eos_value=self.eos_value,
                pad_value=self.pad_value,
                distanceType=ChamferDistanceType.cross_entropy,
            )
        elif config.with_set_prediction == SetPredictionType.chamfer_unilateral_ce:
            self.criterion = ChamferDistanceCriterion(
                eos_value=self.eos_value,
                pad_value=self.pad_value,
                distanceType=ChamferDistanceType.unilateral_cross_entropy,
            )
        elif config.with_set_prediction == SetPredictionType.pooled_bce:
            self.criterion = PooledBinaryCrossEntropy(
                eos_value=self.eos_value, pad_value=self.pad_value
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.pad_value, reduction="mean"
            )

    @staticmethod
    def _create_decoder(config: IngredientPredictorVITConfig, vocab_size: int):
        decoder_layers = []
        for i in range(config.layers):
            decoder_layers.extend(
                [
                    nn.Conv1d(
                        config.embed_size, config.embed_size, kernel_size=1, bias=False,
                    ),
                    nn.Dropout(config.dropout),
                    nn.BatchNorm1d(config.embed_size),
                    nn.ReLU(),
                ]
            )
        decoder_layers.append(
            nn.Conv1d(config.embed_size, vocab_size - 1, kernel_size=1, bias=False,),
        )
        return nn.Sequential(*decoder_layers)

    def _forward_impl(
        self,
        img_features: torch.Tensor,
        label_target: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        assert self.max_num_ingredients + 1 == img_features.size(-1)
        label_logits = self.decoder(img_features)
        label_logits = label_logits.permute(0, 2, 1)

        if compute_predictions:
            predictions = self._compute_predictions(label_logits)
        else:
            predictions = None

        losses: Dict[str, torch.Tensor] = {}
        if compute_losses:
            if self.permutation_invariant:
                losses = self.criterion(label_logits, label_target)
            else:
                # Flatten the prediction and do a simple cross-entropy loss
                label_logits_v = label_logits.reshape(
                    label_logits.size(0) * label_logits.size(1), -1
                )
                label_target_v = label_target.view(-1)  # TODO - padding value?
                loss = self.criterion(label_logits_v, label_target_v)
                losses["label_loss"] = loss

        return losses, predictions

    def _compute_predictions(self, label_logits: torch.Tensor):
        """

        """
        predictions = []
        label_logits = label_logits.clone().detach()
        for i in range(label_logits.shape[1]):
            el = label_logits[:, i, :]
            # predicted mask
            if i == 0:
                predicted_mask = torch.zeros(el.shape).type_as(el)
            else:
                batch_ind = [
                    j
                    for j in range(el.shape[0])
                    if predictions[i - 1][j] != self.eos_value
                ]
                predictions_new = predictions[i - 1][batch_ind]
                predicted_mask[batch_ind, predictions_new] = float("-inf")

            # mask previously selected ids
            el += predicted_mask
            _, predicted = el.max(1)
            predicted = predicted.detach()
            predictions.append(predicted)
        predictions = torch.stack(predictions, 1)
        # mask labels after finding eos (cardinality)
        sample_mask = mask_from_eos(
            predictions, eos_value=self.eos_value, mult_before=False
        )
        predictions[sample_mask == 0] = self.pad_value
        return predictions
