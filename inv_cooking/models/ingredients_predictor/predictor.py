# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
# Code adapted from image-to-set
# Copyright (c) Facebook, Inc. and its affiliates.


import abc
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class IngredientsPredictor(nn.Module, abc.ABC):
    """
    Interface of any ingredient predictor implementation
    """

    def __init__(self, requires_eos: bool):
        super().__init__()
        self.requires_eos = requires_eos

    def forward(
        self,
        img_features: torch.Tensor,
        label_target: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict the ingredients of the image features extracted by an image encoder
        :param img_features: image features - shape (N, embedding_size, sequence_length)
        :param label_target: ground truth, the ingredients to find - shape (N, max_num_ingredients)
        :param compute_losses: whether or not to compute loss between output and target
        :param compute_predictions: whether or not to output the predicted ingredients
        """
        assert (label_target is not None and compute_losses) or (
            label_target is None and not compute_losses
        )

        losses: Dict[str, torch.Tensor] = {}
        predictions: Optional[torch.Tensor] = None
        if not compute_losses and not compute_predictions:
            return losses, predictions
        else:
            return self._forward_impl(
                img_features, label_target, compute_losses, compute_predictions
            )

    @abc.abstractmethod
    def _forward_impl(
        self,
        img_features: torch.Tensor,
        label_target: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        ...
