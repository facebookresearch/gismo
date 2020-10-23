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
                 decoder,
                 maxnumlabels,
                 embed_size,
                 dropout_encoder,
                 crit=None,
                 crit_eos=None,
                 crit_cardinality=None,
                 pad_value=0,
                 perminv=True,
                 is_decoder_ff=False,
                 th=0.5,
                 loss_label='bce',
                 replacement=False,  # this is set to False because it is the ingredient prediction model
                 card_type='none',
                 dataset='recipe1m',
                 U=2.36,
                 use_empty_set=False,
                 eps=1e-8):

        super(Im2Ingr, self).__init__()

        self.image_encoder = ImageEncoder() # args.embed_size, args.dropout_encoder,  args.image_model)

        self.ingr_predictor = get_ingr_predictor({}, 1)
        # decoder,
        # args.maxnumlabels,
        # crit=label_loss,
        # crit_eos=eos_loss,
        # crit_cardinality=cardinality_loss,
        # pad_value=pad_value,
        # perminv=args.perminv,
        # is_decoder_ff=True if args.decoder == 'ff' else False,
        # th=args.th,
        # loss_label=args.label_loss,
        # card_type=args.pred_cardinality,
        # dataset=args.dataset,
        # U=args.U)

    def forward(self, img, label_target=None, maxnumlabels=0, compute_losses=False, compute_predictions=False):

        img_features = self.image_encoder(img)

        losses, predictions = self.ingr_predictor(img_features, label_target=None, 
                                                  maxnumlabels=maxnumlabels, 
                                                  compute_losses=compute_losses, 
                                                  compute_predictions=compute_predictions)

        return losses, predictions

