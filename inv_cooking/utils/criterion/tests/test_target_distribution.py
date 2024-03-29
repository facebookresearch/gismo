# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from inv_cooking.utils.criterion.target_distribution import (
    TargetDistributionCriterion,
    _to_target_distribution,
)


def test_target_distribution_construction():
    targets = torch.LongTensor(
        [
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 0, 0],
        ]
    ).cuda()
    distribution = _to_target_distribution(targets, epsilon=1e-8)
    expected = torch.tensor(
        [
            [1.0000, 0.0000, 0.0000, 0.0000],
            [0.5000, 0.0000, 0.5000, 0.0000],
            [0.2500, 0.2500, 0.2500, 0.2500],
        ]
    )
    assert torch.allclose(distribution.cpu(), expected, atol=1e-4)


def test_target_distribution_loss():
    logits = torch.FloatTensor(
        [
            [12, 15, -10, 3],
            [17, -50, 30, -10],
        ]
    )
    targets = torch.LongTensor(
        [
            [1, 0, 0, 0],
            [1, 0, 1, 0],
        ]
    )

    td = TargetDistributionCriterion(reduction="none")
    out = td(logits, targets)
    assert torch.allclose(out, torch.tensor([3.0486, 6.5000]), atol=1e-4)

    td = TargetDistributionCriterion(reduction="mean")
    out = td(logits, targets)
    assert torch.allclose(out, torch.tensor(4.7743), atol=1e-4)


def test_target_distribution_loss_relative_comparison():
    targets = torch.LongTensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
        ]
    )
    logits_good = torch.FloatTensor(
        [
            [-10, -10, -10, -10],
            [10, -10, -10, -10],
            [10, -10, 10, -10],
            [10, 10, 10, 10],
        ]
    )
    logits_bad = -1 * logits_good

    td = TargetDistributionCriterion(reduction="none")
    loss_good = td(logits_good, targets)
    loss_bad = td(logits_bad, targets)
    assert (loss_good <= loss_bad).sum() == targets.size(0)
    print(loss_good)
    print(loss_bad)
