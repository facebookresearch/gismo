# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from inv_cooking.config import ImageEncoderConfig
from inv_cooking.models.image_encoder import ImageEncoder
from inv_cooking.models.ingredients_predictor import get_ingr_predictor


class Im2Ingr(nn.Module):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: DictConfig,
        ingr_vocab_size: int,
        dataset_name: str,
        max_num_labels: int,
        ingr_eos_value,
    ):
        super().__init__()
        self.ingr_vocab_size = ingr_vocab_size

        if ingr_pred_config.freeze:
            image_encoder_config.freeze = "all"

        self.image_encoder = ImageEncoder(
            ingr_pred_config.embed_size, image_encoder_config
        )

        self.ingr_predictor = get_ingr_predictor(
            ingr_pred_config,
            vocab_size=ingr_vocab_size,
            dataset=dataset_name,
            maxnumlabels=max_num_labels,
            eos_value=ingr_eos_value,
        )

    def forward(
        self,
        image: torch.Tensor,
        label_target: Optional[torch.Tensor] = None,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Predict a set of ingredients from an image
        :param image: input image - shape is (N, C, H, W)
        :param label_target: ground truth of ingredients to predict - shape is (N, max_num_labels)
        :param compute_losses: whether or not to compute the loss
        :param compute_predictions: whether or not to output the ingredients prediction
        """
        img_features = self.image_encoder(image)
        return self.ingr_predictor(
            img_features,
            label_target=label_target,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
