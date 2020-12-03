import random
import numpy as np

import torch
import torch.nn as nn

from inv_cooking.models.modules.transformer_decoder import DecoderTransformer


class RecipeGenerator(nn.Module):
    def __init__(self, args,
                 instr_vocab_size,
                 maxrecipelen):

        super(RecipeGenerator, self).__init__()

        # recipe (text) generation model
        self.model = DecoderTransformer(args.embed_size, instr_vocab_size,
                                        dropout=args.dropout, 
                                        seq_length=maxrecipelen,
                                        attention_nheads=args.n_att_heads, 
                                        pos_embeddings=True,
                                        num_layers=args.layers,
                                        normalize_before=args.normalize_before)

        # recipe loss
        self.crit = nn.CrossEntropyLoss(ignore_index=instr_vocab_size-1, reduction='mean')
        # MaskedCrossEntropyCriterion(ignore_index=[instr_vocab_size-1], reduce=False)

    def forward(self, img_features, ingr_features, ingr_mask, recipe, compute_losses=False, compute_predictions=False, greedy=True, temperature=1.0):

        loss = None
        predictions = None

        if compute_losses:
            output_logits, _ = self.model(features=ingr_features, 
                                          mask=ingr_mask, 
                                          captions=recipe, 
                                          other_features=img_features)

            # compute loss
            output_logits = output_logits[:, :-1, :].contiguous()
            output_logits = output_logits.view(output_logits.size(0) * output_logits.size(1), -1)
            targets = recipe[:, 1:]
            targets = targets.contiguous().view(-1)
            loss = self.crit(output_logits, targets)

        if compute_predictions:
            predictions, _ = self.model.sample(features=ingr_features, 
                                               mask=ingr_mask, 
                                               other_features=img_features,
                                               greedy=greedy, 
                                               temperature=temperature, 
                                               first_token_value=0,
                                               replacement=True)
                                               
        return loss, predictions