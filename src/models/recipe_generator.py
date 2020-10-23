import torch
import torch.nn as nn
import random
import numpy as np
from models.image_encoder import ImageEncoder
from models.ingredients_encoder import IngredientsEncoder
from models.modules.transformer_decoder import DecoderTransformer
from models.modules.multihead_attention import MultiheadAttention
from utils.metrics import softIoU, MaskedCrossEntropyCriterion
import pickle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_recipe_generator(args, ingr_vocab_size, instrs_vocab_size):

    # recipe (text) generation model
    model = DecoderTransformer(args.embed_size, instrs_vocab_size,
                                 dropout=args.dropout_decoder_r, 
                                 seq_length=args.maxseqlen,
                                 num_instrs=args.maxnuminstrs,
                                 attention_nheads=args.n_att, 
                                 num_layers=args.transf_layers,
                                 normalize_before=True,
                                 normalize_inputs=False,
                                 last_ln=False,
                                 scale_embed_grad=False)

    # recipe loss
    criterion = MaskedCrossEntropyCriterion(ignore_index=[instrs_vocab_size-1], reduce=False)

    return model, criterion


class RecipeGenerator(nn.Module):
    def __init__(self, recipe_decoder, 
                 crit=None,
                 pad_value=0):

        super(RecipeGenerator, self).__init__()

        self.recipe_decoder = recipe_decoder
        self.crit = crit
        self.pad_value = pad_value

    def forward(self, img_features, ingr_features, ingr_mask, recipe, compute_losses=False, compute_predictions=False, greedy=True, temperature=1.0):

        losses = None
        predictions = None

        if compute_losses:

            targets = recipe[:, 1:]
            targets = targets.contiguous().view(-1)

            output_logits, _ = self.recipe_decoder(ingr_features, ingr_mask, recipe, other_features=img_features)
            output_logits = output_logits[:, :-1, :].contiguous()
            output_logits = output_logits.view(outputs.size(0) * output_logits.size(1), -1)

            losses = self.crit(output_logits, targets)

        if compute_predictions:
            predictions, _ = self.recipe_decoder.sample(ingr_features, ingr_mask, greedy, temperature, img_features, first_token_value=0)

        return loss, predictions