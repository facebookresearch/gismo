# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from inv_cooking.training.utils import _BaseModule


def test_checkpoint_to_cpu():
    model = nn.Linear(100, 10).cuda()
    fake_state = torch.randn(size=(200, 10)).cuda()
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state": {"grads": [fake_state]},
    }

    _BaseModule.recursively_move_to_cpu(checkpoint)
    for param_name in checkpoint["state_dict"].keys():
        assert model.state_dict()[param_name].is_cuda
        assert not checkpoint["state_dict"][param_name].is_cuda
    assert fake_state.is_cuda
    assert not checkpoint["optimizer_state"]["grads"][0].is_cuda
