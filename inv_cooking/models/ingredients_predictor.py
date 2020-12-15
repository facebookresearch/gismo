# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import cast

import numpy as np
import torch
import torch.nn as nn

from inv_cooking.config import (
    IngredientPredictorConfig,
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
)
from inv_cooking.models.modules.ff_decoder import FFDecoder
from inv_cooking.models.modules.rnn_decoder import DecoderRNN
from inv_cooking.models.modules.transformer_decoder import DecoderTransformer
from inv_cooking.models.modules.utils import freeze_fn
from inv_cooking.utils.metrics import DC, DCLoss, SoftIoULoss, TargetDistributionLoss


def label2_k_hots(labels, pad_value, remove_eos=False):

    # input labels to one hot vector
    inp_ = torch.unsqueeze(labels, 2)
    k_hots = torch.zeros(labels.size(0), labels.size(1), pad_value + 1).type_as(inp_)
    k_hots.scatter_(2, inp_, 1)
    k_hots, _ = k_hots.max(dim=1)

    # remove pad position
    k_hots = k_hots[:, :-1]

    # handle eos
    if remove_eos:
        # this is used by tfset/lstmset when computing losses and
        # by all auto-regressive models when computing f1 metrics
        k_hots = k_hots[:, 1:]
    return k_hots


def mask_from_eos(prediction, eos_value, mult_before=True):
    mask = torch.ones(prediction.size()).type_as(prediction).byte()
    mask_aux = torch.ones(prediction.size(0)).type_as(prediction).byte()

    # find eos in label prediction
    for idx in range(prediction.size(1)):
        # force mask to have 1s in the first position to avoid division
        # by 0 when predictions start with eos
        if idx == 0:
            continue
        if mult_before:
            mask[:, idx] = mask[:, idx] * mask_aux
            mask_aux = mask_aux * (prediction[:, idx] != eos_value)
        else:
            mask_aux = mask_aux * (prediction[:, idx] != eos_value)
            mask[:, idx] = mask[:, idx] * mask_aux
    return mask


def predictions_to_idxs(
    label_logits,
    maxnumlabels,  ## TODO check how this is used, and whether the +1 in the set predictor construction is necessary
    pad_value,
    th=1,
    cardinality_prediction=None,
    which_loss="bce",
    accumulate_probs=False,
    use_empty_set=False,
):

    assert th > 0 and th <= 1

    card_offset = 0 if use_empty_set else 1

    # select topk elements
    probs, idxs = torch.topk(
        label_logits, k=maxnumlabels, dim=1, largest=True, sorted=True
    )
    idxs_clone = idxs.clone()

    # mask to identify elements within the top-maxnumlabel ones which satisfy the threshold th
    if which_loss == "td":
        # cumulative threshold
        mask = torch.ones(probs.size()).type_as(probs).byte()
        for idx in range(probs.size(1)):
            mask_step = torch.sum(probs[:, 0:idx], dim=-1) < th
            mask[:, idx] = mask[:, idx] * mask_step
    else:
        # probility threshold
        mask = (probs > th).byte()

    # if the model has cardinality prediction
    if cardinality_prediction is not None:

        # get the argmax for each element in the batch to get the cardinality
        # (note that the output is N - 1, e.g. argmax = 0 means that there's 1 element)
        # unless we are in the empty set case, e.g. argmax = 0 means there there are 0 elements

        if accumulate_probs:
            for c in range(cardinality_prediction.size(-1)):
                value = torch.sum(torch.log(probs[:, 0 : c + 1]), dim=-1)
                cardinality_prediction[:, c] += value

        # select cardinality
        _, card_idx = torch.max(cardinality_prediction, dim=-1)

        mask = torch.ones(probs.size()).type_as(probs).byte()
        aux_mask = torch.ones(mask.size(0)).type_as(probs).byte()

        for i in range(mask.size(-1)):
            # If the cardinality prediction is higher than i, it means that from this point
            # on the mask must be 0. Predicting 0 cardinality means 0 objects when
            # use_empty_set=True and 1 object when use_empty_set=False
            # real cardinality value is
            above_cardinality = i < card_idx + card_offset
            # multiply the auxiliar mask with this condition
            # (once you multiply by 0, the following entries will also be 0)
            aux_mask = aux_mask * above_cardinality
            mask[:, i] = aux_mask
    else:
        if not use_empty_set:
            mask[:, 0] = 1

    idxs_clone[mask == 0] = pad_value

    return idxs_clone


def get_ingr_predictor(
    config: IngredientPredictorConfig,
    vocab_size: int,
    dataset: str,
    maxnumlabels: int,
    eos_value: int,
):
    """
    Create the ingredient predictor based on the configuration
    """

    cardinality_pred = config.cardinality_pred

    # build ingredients predictor
    if "ff" in config.model:
        config = cast(IngredientPredictorFFConfig, config)
        print(
            "Building feed-forward decoder {}. Embed size {} / Dropout {} / "
            " Max. Num. Labels {} / Num. Layers {}".format(
                config.model,
                config.embed_size,
                config.dropout,
                maxnumlabels,
                config.layers,
            ),
            flush=True,
        )
        decoder = FFDecoder(
            config.embed_size,
            vocab_size,
            config.embed_size,
            dropout=config.dropout,
            pred_cardinality=cardinality_pred,
            nobjects=maxnumlabels,
            n_layers=config.layers,
        )

    elif "lstm" in config.model:
        config = cast(IngredientPredictorLSTMConfig, config)
        maxnumlabels += 1  ## TODO: check +1 (required for EOS token)
        print(
            "Building LSTM decoder {}. Embed size {} / Dropout {} / Max. Num. Labels {}. ".format(
                config.model, config.embed_size, config.dropout, maxnumlabels
            ),
            flush=True,
        )
        decoder = DecoderRNN(
            config.embed_size,
            config.embed_size,
            vocab_size,
            dropout=config.dropout,
            seq_length=maxnumlabels,
        )

    elif "tf" in config.model:
        config = cast(IngredientPredictorTransformerConfig, config)
        maxnumlabels += 1  ## TODO: check +1 (required for EOS token)
        print(
            "Building Transformer decoder {}. Embed size {} / Dropout {} / Max. Num. Labels {} / "
            "Num. Attention Heads {} / Num. Layers {}.".format(
                config.model,
                config.embed_size,
                config.dropout,
                maxnumlabels,
                config.n_att,
                config.layers,
            ),
            flush=True,
        )
        decoder = DecoderTransformer(
            config.embed_size,
            vocab_size,
            dropout=config.dropout,
            seq_length=maxnumlabels,
            attention_nheads=config.n_att,
            pos_embeddings=False,
            num_layers=config.layers,
            learned=False,
            normalize_before=True,
        )

    # label and eos loss
    label_losses = {
        "bce": nn.BCEWithLogitsLoss(reduction="mean")
        if "ff" in config.model
        else nn.BCELoss(reduction="mean"),
        "iou": SoftIoULoss(reduction="mean"),
        "td": TargetDistributionLoss(reduction="mean"),
    }
    pad_value = vocab_size - 1

    if "ff" in config.model:
        loss_key = {k for k in label_losses.keys() if k in config.model}.pop()
        label_loss = label_losses[loss_key]
        eos_loss = None
    elif config.with_set_prediction:
        loss_key = "bce"
        label_loss = label_losses["bce"]
        eos_loss = nn.BCELoss(reduction="none")
    else:
        loss_key = "cross-entropy"
        label_loss = nn.CrossEntropyLoss(ignore_index=pad_value, reduction="mean")
        eos_loss = None

    # cardinality loss
    if cardinality_pred == "dc":
        print("Using Dirichlet-Categorical cardinality loss.", flush=True)
        cardinality_loss = DCLoss(U=config.U, dataset=dataset, reduction="mean")
    elif cardinality_pred == "cat":
        print("Using categorical cardinality loss.", flush=True)
        cardinality_loss = nn.CrossEntropyLoss(reduction="mean")
    else:
        print("Using no cardinality loss.", flush=True)
        cardinality_loss = None

    model = IngredientsPredictor(
        decoder,
        maxnumlabels,
        vocab_size,
        crit=label_loss,
        crit_eos=eos_loss,
        crit_cardinality=cardinality_loss,
        pad_value=pad_value,
        eos_value=eos_value,
        perminv=config.with_set_prediction,
        is_decoder_ff="ff" in config.model,
        loss_label=loss_key,
        card_type=cardinality_pred,
        dataset=dataset,
    )

    if config.freeze:
        freeze_fn(model)
    return model


class IngredientsPredictor(nn.Module):
    def __init__(
        self,
        decoder,
        maxnumlabels,
        vocab_size,
        crit=None,
        crit_eos=None,
        crit_cardinality=None,
        pad_value=0,
        eos_value=0,
        perminv=True,
        is_decoder_ff=False,
        th=0.5,
        loss_label="bce",
        replacement=False,  # this is set to False because it is the ingredient prediction model
        card_type="none",
        dataset="recipe1m",
        U=2.36,
        use_empty_set=False,
        eps=1e-8,
    ):

        super(IngredientsPredictor, self).__init__()
        self.decoder = decoder
        self.is_decoder_ff = is_decoder_ff
        self.maxnumlabels = maxnumlabels
        self.crit = crit
        self.th = th
        self.perminv = perminv
        self.pad_value = pad_value
        self.eos_value = eos_value
        self.crit_eos = crit_eos
        self.crit_cardinality = crit_cardinality
        self.loss_label = loss_label
        self.replacement = replacement
        self.card_type = card_type
        self.dataset = dataset  # TODO: this may not be used at all in this code
        self.u_term = math.log(U)  # TODO: this may not be used at all in this code
        self.eps = eps
        self.use_empty_set = (
            use_empty_set  # TODO: this may not be used at all in this code
        )
        self.vocab_size = vocab_size

    def forward(
        self,
        img_features,
        label_target=None,
        compute_losses=False,
        compute_predictions=False,
    ):

        losses = {}
        predictions = None

        assert (label_target is not None and compute_losses) or (
            label_target is None and not compute_losses
        )

        if not compute_losses and not compute_predictions:
            return losses, predictions

        if self.is_decoder_ff:
            # use ff decoder to predict set of labels and cardinality
            label_logits, cardinality_logits = self.decoder(img_features)

            if compute_losses:
                # label target to k_hot
                target_k_hot = label2_k_hots(label_target, self.pad_value)
                target_k_hot = target_k_hot.type_as(label_logits)
                # cardinality target
                cardinality_target = target_k_hot.sum(dim=-1).unsqueeze(1)

                # compute labels loss
                losses["label_loss"] = self.crit(label_logits, target_k_hot)

                # compute cardinality loss if needed
                if self.crit_cardinality is not None:
                    # subtract 1 from num_target to match class idxs (1st label corresponds to class 0) only
                    # 1st label corresponds to 0 only if use_empty_set is false
                    # otherwise, 1st label corresponds to 1
                    offset = 0 if self.use_empty_set else 1
                    losses["cardinality_loss"] = self.crit_cardinality(
                        cardinality_logits,
                        (cardinality_target.squeeze() - offset).long(),
                    )

            if compute_predictions:
                # consider cardinality
                if self.card_type == "dc" and self.loss_label == "bce":
                    offset = 0 if self.use_empty_set else 1
                    cardinality = torch.log(
                        DC(cardinality_logits, dataset=self.dataset)
                    )
                    u_term = np.array(list(range(cardinality.size(-1)))) + offset
                    u_term = u_term * self.u_term
                    u_term = torch.from_numpy(u_term).unsqueeze(0).type_as(cardinality)
                    cardinality = cardinality + u_term
                elif self.card_type == "cat":
                    cardinality = torch.nn.functional.log_softmax(
                        cardinality_logits + self.eps, dim=-1
                    )
                else:
                    cardinality = None

                # apply nonlinearity to label logits
                if self.loss_label == "td":
                    label_probs = nn.functional.softmax(label_logits, dim=-1)
                else:
                    label_probs = torch.sigmoid(label_logits)

                # get label ids
                predictions = predictions_to_idxs(
                    label_probs,
                    self.maxnumlabels,
                    self.pad_value,
                    th=self.th,
                    cardinality_prediction=cardinality,
                    which_loss=self.loss_label,
                    accumulate_probs=self.card_type == "dc"
                    and self.loss_label == "bce",
                    use_empty_set=self.use_empty_set,
                )

        else:  # auto-regressive models

            # use auto-regressive decoder to predict labels (sample function)
            # output label_logits is only used to compute losses in case of self.perminv (no teacher forcing)
            # predictions output is used for all auto-regressive models
            predictions, label_logits = self.decoder.sample(
                img_features, None, first_token_value=0, replacement=self.replacement
            )

            if compute_predictions:
                # mask labels after finding eos (cardinality)
                sample_mask = mask_from_eos(
                    predictions, eos_value=self.eos_value, mult_before=False
                )
                predictions[sample_mask == 0] = self.pad_value

            if compute_losses:
                # add dummy first word to sequence and remove last
                first_word = torch.zeros(len(label_target)).type_as(label_target)
                shift_target = torch.cat([first_word.unsqueeze(-1), label_target], -1)[
                    :, :-1
                ]
                if self.perminv:
                    # autoregressive mode for decoder when training with permutation invariant objective
                    # e.g. lstmset and tfset

                    # apply softmax nonlinearity before pooling across timesteps
                    label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

                    # find idxs for eos label
                    # eos probability is the one assigned to the position self.eos_value of the softmax
                    # this is used with bce loss only
                    eos = label_probs[:, :, self.eos_value]
                    eos_pos = (
                        label_target == self.eos_value
                    )  # all zeros except position where eos is in the gt
                    eos_head = (label_target != self.pad_value) & (
                        label_target != self.eos_value
                    )  # 1s for gt label positions, 0s starting from eos position in the gt
                    eos_target = (
                        ~eos_head
                    )  # 0s for gt label positions, 1s starting from eos position in the gt

                    # select transformer steps to pool (steps corresponding to set elements, i.e. labels)
                    label_probs = label_probs * eos_head.float().unsqueeze(-1)

                    # pool
                    label_probs, _ = torch.max(label_probs, dim=1)

                    # compute label loss
                    target_k_hot = label2_k_hots(
                        label_target, self.pad_value, remove_eos=True
                    )
                    loss = self.crit(label_probs[:, 1:], target_k_hot.float())
                    losses["label_loss"] = loss

                    # compute eos loss
                    eos_loss = self.crit_eos(eos, eos_target.float())
                    # eos loss is computed for all timesteps <= eos in gt and
                    # equally penalizes the head (all 0s) and the true eos position (1)
                    losses["eos_loss"] = (
                        0.5
                        * (eos_loss * eos_pos.float()).sum(1)
                        / (eos_pos.float().sum(1) + self.eps)
                        + 0.5
                        * (eos_loss * eos_head.float()).sum(1)
                        / (eos_head.float().sum(1) + self.eps)
                    ).mean()

                else:
                    # other autoregressive models
                    # we need to recompute logits using teacher forcing (forward pass)
                    label_logits, _ = self.decoder(img_features, None, shift_target)
                    label_logits_v = label_logits.view(
                        label_logits.size(0) * label_logits.size(1), -1
                    )

                    # compute label loss
                    label_target_v = label_target.view(-1)
                    loss = self.crit(label_logits_v, label_target_v)
                    losses["label_loss"] = loss

        return losses, predictions
