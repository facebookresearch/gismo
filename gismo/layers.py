# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dgl.function as fn
import torch.nn as nn


class GCNConv(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata["h"] = self.linear(x)
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum(msg="m", out="h"))
            return g.ndata["h"]
