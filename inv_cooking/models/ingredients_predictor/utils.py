# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


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
    th:float=1,
    cardinality_prediction=None,
    which_loss="bce",
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