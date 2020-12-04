# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

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
        maxnumlabels: int,
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
            maxnumlabels=maxnumlabels,
            eos_value=ingr_eos_value,
        )

    def forward(
        self, img, label_target=None, compute_losses=False, compute_predictions=False
    ):
        img_features = self.image_encoder(img)
        losses, predictions = self.ingr_predictor(
            img_features,
            label_target=label_target,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        return losses, predictions
