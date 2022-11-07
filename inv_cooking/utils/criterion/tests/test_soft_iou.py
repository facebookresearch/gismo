# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from inv_cooking.utils.criterion.soft_iou import SoftIoUCriterion, soft_iou


def test_soft_iou():
    logits = torch.FloatTensor(
        [
            [12, 15, -10],
            [17, -50, 30],
        ]
    )
    targets = torch.LongTensor(
        [
            [1, 0, 0],
            [1, 0, 1],
        ]
    )

    out = soft_iou(logits, targets)
    expected = torch.tensor([[0.5], [1.0]])
    assert torch.allclose(out, expected, atol=1e-4)

    criterion = SoftIoUCriterion(reduction="none")
    expected = torch.tensor([[0.5], [0.0]])
    losses = criterion(logits, targets)
    assert torch.allclose(losses, expected, atol=1e-4)

    criterion = SoftIoUCriterion(reduction="mean")
    losses = criterion(logits, targets)
    assert torch.allclose(losses, torch.tensor(0.25), atol=1e-4)
