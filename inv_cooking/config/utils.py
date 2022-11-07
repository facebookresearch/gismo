# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import field

from omegaconf import DictConfig


def untyped_config():
    return field(default_factory=lambda: DictConfig(content=dict()))
