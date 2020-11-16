# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class MaskedCrossEntropyCriterion(_WeightedLoss):

    def __init__(self, ignore_index=[-100], reduce=None):
        super(MaskedCrossEntropyCriterion, self).__init__()
        self.padding_idx = ignore_index
        self.reduce = reduce

    def forward(self, outputs, targets):
        lprobs = nn.functional.log_softmax(outputs, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        for idx in self.padding_idx:
            # remove padding idx from targets to allow gathering without error (padded entries will be suppressed later)
            targets[targets == idx] = 0

        nll_loss = -lprobs.gather(dim=-1, index=targets.unsqueeze(1))
        if self.reduce:
            nll_loss = nll_loss.sum()

        return nll_loss.squeeze()

def DC(alphas, dataset='recipe1m'):

    if dataset == 'recipe1m':
        cms = np.array([
            0.0000e+00, 1.1570e+04, 2.7383e+04, 4.5583e+04, 6.2422e+04, 7.2228e+04, 7.6909e+04,
            7.6319e+04, 7.0102e+04, 5.9216e+04, 4.6648e+04, 3.4192e+04, 2.3283e+04, 1.5590e+04,
            1.0079e+04, 6.2400e+03, 3.7690e+03, 2.2110e+03, 1.3430e+03, 2.7000e+01
        ])

    cms = cms[0:alphas.size(-1)]
    c = sum(cms)

    cms = torch.from_numpy(cms).type_as(alphas)
    cms = cms.unsqueeze(0)

    num = alphas + cms
    den = (torch.sum(alphas, dim=-1) + c).unsqueeze(1)
    dc_alphas = num / den

    return dc_alphas


class DCLoss(nn.Module):

    def __init__(self, U, dataset, reduction='none', e=1e-8):
        super(DCLoss, self).__init__()
        assert reduction in ['none', 'mean']
        self.U = math.log(U)
        self.dataset = dataset
        self.offset = 0 if self.dataset in ['coco', 'nuswide'] else 1
        self.reduction = reduction
        self.e = e

    def forward(self, input, target):
        loss = nn.NLLLoss(reduction='none')(
            torch.log(DC(input, dataset=self.dataset) + self.e),
            target) - ((target + self.offset) * self.U).float()

        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


def softIoU(out, target, sum_axis=1, e=1e-8):
    # logits to probs
    out = torch.sigmoid(out)
    # loss
    num = (out * target).sum(sum_axis, True) + e
    den = (out + target - out * target).sum(sum_axis, True) + e
    iou = num / den

    return iou


class softIoULoss(nn.Module):

    def __init__(self, reduction='none', e=1e-8):
        super(softIoULoss, self).__init__()
        assert reduction in ['none', 'mean']
        self.reduction = reduction
        self.e = e

    def forward(self, inputs, targets):
        loss = 1.0 - softIoU(inputs, targets, e=self.e)

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class targetDistLoss(nn.Module):

    def __init__(self, reduction='none', e=1e-8):
        super(targetDistLoss, self).__init__()
        assert reduction in ['none', 'mean']
        self.reduction = reduction
        self.e = e

    def forward(self, label_prediction, label_target):
        # create target distribution
        # check if the target is all 0s
        cardinality_target = label_target.sum(dim=-1).unsqueeze(1)
        is_empty = (cardinality_target == 0).float()

        # create flat distribution
        flat_target = 1 / label_target.size(-1)
        flat_target = torch.FloatTensor(
                        np.array(flat_target)).unsqueeze(-1).unsqueeze(-1).type_as(cardinality_target)

        # divide target by number of elements and add equal prob to all elements for the empty sets
        target_distribution = label_target.float() / (
            cardinality_target + self.e) + is_empty * flat_target

        # loss
        loss = target_distribution * torch.nn.functional.log_softmax(label_prediction, dim=-1)
        loss = -torch.sum(loss, dim=-1)

        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


def update_error_counts(error_counts, y_pred, y_true, which_metrics):

    if 'o_f1' in which_metrics:
        error_counts['tp_all'] += (y_pred * y_true).sum().item()
        error_counts['fp_all'] += (y_pred * (1 - y_true)).sum().item()
        error_counts['fn_all'] += ((1 - y_pred) * y_true).sum().item()

    if 'c_f1' in which_metrics:
        error_counts['tp_c'] += (y_pred * y_true).sum(0).cpu().data.numpy()
        error_counts['fp_c'] += (y_pred * (1 - y_true)).sum(0).cpu().data.numpy()
        error_counts['fn_c'] += ((1 - y_pred) * y_true).sum(0).cpu().data.numpy()
        error_counts['tn_c'] += ((1 - y_pred) * (1 - y_true)).sum(0).cpu().data.numpy()

    if 'i_f1' in which_metrics:
        error_counts['tp_i'] = (y_pred * y_true).sum(1).cpu().data.numpy()
        error_counts['fp_i'] = (y_pred * (1 - y_true)).sum(1).cpu().data.numpy()
        error_counts['fn_i'] = ((1 - y_pred) * y_true).sum(1).cpu().data.numpy()   


def compute_metrics(error_counts, which_metrics, eps=1e-8, weights=None):

    ret_metrics = {}

    if 'o_f1' in which_metrics:
        pre = (error_counts['tp_all'] + eps) / (error_counts['tp_all'] + error_counts['fp_all'] + eps)
        rec = (error_counts['tp_all'] + eps) / (error_counts['tp_all'] + error_counts['fn_all'] + eps)

        o_f1 = 2 * (pre * rec) / (pre + rec)
        ret_metrics['o_f1'] = o_f1

    if 'c_f1' in which_metrics:
        pre = (error_counts['tp_c'] + eps) / (error_counts['tp_c'] + error_counts['fp_c'] + eps)
        rec = (error_counts['tp_c'] + eps) / (error_counts['tp_c'] + error_counts['fn_c'] + eps)

        f1_perclass = 2 * (pre * rec) / (pre + rec)
        f1_perclass_avg = np.average(f1_perclass, weights=weights)
        ret_metrics['c_f1'] = f1_perclass_avg

    if 'i_f1' in which_metrics:
        pre = (error_counts['tp_i'] + eps) / (error_counts['tp_i'] + error_counts['fp_i'] + eps)
        rec = (error_counts['tp_i'] + eps) / (error_counts['tp_i'] + error_counts['fn_i'] + eps)

        f1_i = 2 * (pre * rec) / (pre + rec)  ## TODO: check this should be of size batch_sizex1
        ret_metrics['i_f1'] = f1_i    

    return ret_metrics
