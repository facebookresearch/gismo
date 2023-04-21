# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Code adapted from https://github.com/facebookresearch/image-to-set
# This source code is licensed under the MIT license found in the
# LICENSE file in https://github.com/facebookresearch/image-to-set

from typing import Optional

import torch

from inv_cooking.config import IngredientPredictorCriterion


def label2_k_hots(labels: torch.Tensor, pad_value: int, remove_eos: bool = False):
    """
    :param labels: predictions of a model - shape (batch_size, max_num_labels)
    :param pad_value: last valid value of the vocabulary, which represents padding
    :param remove_eos: remove the end of sequence token from the predictions (value 0)
    :return shape
        (batch_size, pad_value + 1) with at most max_num_labels values set to 1 if remove_eos is false
        (batch_size, pad_value) with at most max_num_labels values set to 1 if remove_eos is true
    """

    # input labels to one hot vector
    batch_size, max_num_labels = labels.shape
    labels = torch.unsqueeze(labels, dim=-1)
    k_hots = torch.zeros(batch_size, max_num_labels, pad_value + 1).type_as(labels)
    k_hots.scatter_(dim=2, index=labels, value=1)
    k_hots, _ = k_hots.max(dim=1)

    # remove pad position (padding value is supposed to be vocab_size - 1)
    k_hots = k_hots[:, :-1]

    # handle eos (eos is supposed to be 0)
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


def predictions_to_indices(
    label_probs: torch.Tensor,
    max_num_labels: int,
    pad_value: int,
    threshold: float = 1,
    cardinality_prediction: Optional[torch.Tensor] = None,
    which_loss: IngredientPredictorCriterion = IngredientPredictorCriterion.bce,
) -> torch.Tensor:
    """
    Select the highest logit values, and produce a vector holding the indices of the predicted elements
    :param label_probs: the probabilities for each labels - shape (batch_size, max_num_labels)
    :param max_num_labels: the maximum number of elements that can be predicted
    :param pad_value: padding value (not a real prediction)
    :param threshold: the minimum threshold for an element to be selected
    :param cardinality_prediction: the number of elements to select - shape (batch_size,)
    :param which_loss: the kind of criterion used for the ingredient
    :return the filtered predictions - shape (batch_size, max_num_labels)
    """

    assert 0.0 < threshold <= 1.0

    # select topk elements
    probs, idxs = torch.topk(
        label_probs, k=max_num_labels, dim=1, largest=True, sorted=True
    )
    idxs_clone = idxs.clone()

    # mask to identify elements within the top-max_num_labels ones which satisfy the threshold
    if which_loss == IngredientPredictorCriterion.td:
        # cumulative threshold
        mask = torch.ones(probs.size()).type_as(probs).byte()
        for idx in range(probs.size(1)):
            mask_step = torch.sum(probs[:, 0:idx], dim=-1) < threshold
            mask[:, idx] = mask[:, idx] * mask_step
    else:
        # probability threshold
        mask = (probs > threshold).byte()

    # if the model has cardinality prediction
    if cardinality_prediction is not None:

        # get the argmax for each element in the batch to get the cardinality
        # (note that the output is N - 1, e.g. argmax = 0 means that there's 1 element)
        # unless we are in the empty set case, e.g. argmax = 0 means there there are 0 elements

        # select cardinality
        _, card_idx = torch.max(cardinality_prediction, dim=-1)

        mask = torch.ones(probs.size()).type_as(probs).byte()
        aux_mask = torch.ones(mask.size(0)).type_as(probs).byte()

        for i in range(mask.size(-1)):
            # If the cardinality prediction is higher than i, it means that from this point
            # on the mask must be 0.
            card_offset = 1  # Predicting 0 cardinality means 1 object
            above_cardinality = i < card_idx + card_offset
            # multiply the auxiliar mask with this condition
            # (once you multiply by 0, the following entries will also be 0)
            aux_mask = aux_mask * above_cardinality
            mask[:, i] = aux_mask

    idxs_clone[mask == 0] = pad_value
    return idxs_clone
