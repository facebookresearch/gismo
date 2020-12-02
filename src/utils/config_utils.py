# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os

from omegaconf import OmegaConf

# def get_args_from_json(args, filename):
#     print('Loading args from {}:'.format(filename))
#     config = json.load(open(filename))
#     for key, val in config.items():
#         print(key, flush=True)
#         assert hasattr(args, key)
#         assert isinstance(val, type(getattr(args, key)))
#         setattr(args, key, val)
#     return args


def set_default_ingr_predictor_flags(cfg):
    cfg.ingr_predictor.loss = 'cross-entropy'
    cfg.ingr_predictor.cardinality_pred = 'none'
    cfg.ingr_predictor.type = 'tf'
    cfg.ingr_predictor.perminv = False
    cfg.ingr_predictor.shuffle_labels = False
    cfg.ingr_predictor.replacement = False  ## ingredient prediction


def set_flags_for_ingr_predictor(cfg):
    OmegaConf.set_struct(cfg, False)

    set_default_ingr_predictor_flags(cfg)

    if 'ff_' in cfg.ingr_predictor.model:
        cfg.ingr_predictor.type = 'ff'
        tokens = cfg.ingr_predictor.model.split('_')
        print(tokens)
        assert tokens[1] in ['bce', 'iou', 'td']
        cfg.ingr_predictor.loss = tokens[1]
        if len(tokens) == 3:
            assert tokens[2] in ['dc', 'cat']
            cfg.ingr_predictor.cardinality_pred = tokens[2]
    else:
        cfg.ingr_predictor.type = 'tf' if 'tf' in cfg.ingr_predictor.model else 'lstm'
        cfg.dataset.maxnumlabels += 1 # need an extra position for 'eos' token
        if 'set' in cfg.ingr_predictor.model:
            cfg.ingr_predictor.perminv = True
            cfg.ingr_predictor.loss = 'bce'
        else:
            cfg.ingr_predictor.loss = 'cross-entropy'
            cfg.ingr_predictor.shuffle_labels = True if 'shuffle' in cfg.ingr_predictor.model else False

    OmegaConf.set_struct(cfg, True)
