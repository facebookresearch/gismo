# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn

import os

from models.image_encoder import ImageEncoder
from models.ingredients_predictor import get_ingr_predictor
from models.recipe_generator import get_recipe_generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Im2Ingr(nn.Module):

    def __init__(self,
                 im_args,
                 ingrpred_args,
                 ingr_vocab_size,
                 dataset,
                 maxnumlabels,
                 eps=1e-8):

        super(Im2Ingr, self).__init__()

        self.image_encoder = ImageEncoder(ingrpred_args.embed_size, im_args)

        self.ingr_predictor = get_ingr_predictor(ingrpred_args, vocab_size=ingr_vocab_size, 
                                                 dataset=dataset, maxnumlabels=maxnumlabels)


    def forward(self, img, label_target=None, maxnumlabels=0, compute_losses=False, compute_predictions=False):

        img_features = self.image_encoder(img)

        losses, predictions = self.ingr_predictor(img_features, label_target=None, 
                                                  maxnumlabels=maxnumlabels, 
                                                  compute_losses=compute_losses, 
                                                  compute_predictions=compute_predictions)

        return losses, predictions

