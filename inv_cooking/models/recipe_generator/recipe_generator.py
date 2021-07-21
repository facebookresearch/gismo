import torch
import torch.nn as nn

from inv_cooking.config import RecipeGeneratorConfig
from inv_cooking.models.modules.transformer_decoder import DecoderTransformer
from inv_cooking.utils.criterion import MaskedCrossEntropyCriterion


class RecipeGenerator(nn.Module):
    def __init__(
        self,
        args: RecipeGeneratorConfig,
        instr_vocab_size: int,
        maxrecipelen: int,
        num_cross_attn: int,
    ):
        super().__init__()

        # recipe (text) generation model
        self.model = DecoderTransformer(
            args.embed_size,
            instr_vocab_size,
            dropout=args.dropout,
            seq_length=maxrecipelen,
            attention_nheads=args.n_att_heads,
            pos_embeddings=True,
            num_layers=args.tf_dec_layers,
            activation=args.activation,
            num_cross_attn=num_cross_attn,
        )

        # recipe loss
        self.criterion = MaskedCrossEntropyCriterion(
            ignore_index=instr_vocab_size - 1, reduce=False
        )

    def forward(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        recipe_gt: torch.Tensor,
        compute_losses=False,
        compute_predictions=False,
        greedy=True,
        temperature=1.0,
    ):
        loss = None
        predictions = None

        if compute_losses:
            output_logits, _ = self.model(
                features=features,
                masks=masks,
                captions=recipe_gt,
            )

            # compute loss
            loss = self.criterion(output_logits, recipe_gt[:, 1:])

        if compute_predictions:
            predictions, _ = self.model.sample(
                features=features,
                masks=masks,
                greedy=greedy,
                temperature=temperature,
                first_token_value=0,
                replacement=True,
            )

        return loss, predictions
