# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn

import os

from inv_cooking.models.image_encoder import ImageEncoder
from inv_cooking.models.ingredients_predictor import get_ingr_predictor


class Im2Ingr(nn.Module):

    def __init__(self,
                 im_args,
                 ingrpred_args,
                 ingr_vocab_size,
                 dataset,
                 maxnumlabels,
                 ingr_eos_value,
                 eps=1e-8):

        super(Im2Ingr, self).__init__()

        self.ingr_vocab_size = ingr_vocab_size

        if ingrpred_args.freeze:
            im_args.freeze = 'all'

        self.image_encoder = ImageEncoder(ingrpred_args.embed_size, **im_args)

        self.ingr_predictor = get_ingr_predictor(ingrpred_args, vocab_size=ingr_vocab_size, 
                                                 dataset=dataset, maxnumlabels=maxnumlabels,
                                                 eos_value=ingr_eos_value)



    def forward(self, img, label_target=None, compute_losses=False, compute_predictions=False):

        img_features = self.image_encoder(img)

        losses, predictions = self.ingr_predictor(img_features, label_target=label_target, 
                                                  compute_losses=compute_losses, 
                                                  compute_predictions=compute_predictions)

        return losses, predictions

