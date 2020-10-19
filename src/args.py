# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os


def get_args_from_json(args, filename):
    print('Loading args from {}:'.format(filename))
    config = json.load(open(filename))
    for key, val in config.items():
        print(key, flush=True)
        assert hasattr(args, key)
        assert isinstance(val, type(getattr(args, key)))
        setattr(args, key, val)
    return args


def set_default_model_flags(args):
    setattr(args, 'label_loss', 'cross-entropy')
    setattr(args, 'pred_cardinality', 'none')
    setattr(args, 'decoder', 'tf')
    setattr(args, 'perminv', False)
    setattr(args, 'shuffle_labels', False)
    setattr(args, 'replacement', False)
    return args


def set_flags_for_model(args):
    args = set_default_model_flags(args)
    if 'ff_' in args.model_name:
        setattr(args, 'decoder', 'ff')
        tokens = args.model_name.split('_')
        print(tokens)
        assert tokens[1] in ['bce', 'iou', 'td']
        setattr(args, 'label_loss', tokens[1])
        if len(tokens) == 3:
            assert tokens[2] in ['dc', 'cat']
            setattr(args, 'pred_cardinality', tokens[2])
    else:
        setattr(args, 'decoder', 'tf' if 'tf' in args.model_name else 'lstm')
        if 'set' in args.model_name:
            setattr(args, 'perminv', True)
            setattr(args, 'label_loss', 'bce')
        else:
            setattr(args, 'shuffle_labels', True if 'shuffle' in args.model_name else False)
    return args


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_dir', type=str, default='checkpoints', help='Directory for saving the models')
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='experiment',
        help='Specifies the sub directory for saving the models. The full path will be '
        '<save_dir>/<dataset>/<model_name>/<experiment_name>.')

    parser.add_argument(
        '--dataset',
        type=str,
        default='voc',
        choices=['coco', 'voc', 'nuswide', 'ade20k', 'recipe1m'])
    parser.add_argument(
        '--model_name',
        type=str,
        default='ff_bce',
        choices=[
            'ff_bce', 'ff_iou', 'ff_td', 'ff_bce_dc', 'ff_bce_cat', 'ff_iou_cat', 'ff_td_cat', 'tf',
            'tf_shuffle', 'tfset', 'lstm', 'lstm_shuffle', 'lstmset'
        ])

    # Model parameters
    parser.add_argument(
        '--image_model',
        type=str,
        default='resnet50',
        choices=['resnet50', 'resnet101', 'resnext101_32x8d'])
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--n_att', type=int, default=2)
    parser.add_argument('--tf_layers', type=int, default=1)
    parser.add_argument('--ff_layers', type=int, default=1)

    parser.add_argument(
        '--crop_size', type=int, default=448, help='size for randomly cropping images')
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--log_step', type=int, default=10, help='step size for printing log info')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--scale_learning_rate_cnn', type=float, default=0.01)
    parser.add_argument('--lr_decay_rate', type=float, default=0.99)
    parser.add_argument('--lr_decay_every', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--U', type=float, default=2.36)
    parser.add_argument('--seed', type=int, default=1235)

    parser.add_argument('--dropout_encoder', type=float, default=0.0)
    parser.add_argument('--dropout_decoder', type=float, default=0.0)

    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--maxnumlabels', type=int, default=10)
    parser.add_argument('--finetune_after', type=int, default=0)
    parser.add_argument('--loss_weight', nargs='+', type=float, default=[1.0, 1.0, 0.0])
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument(
        '--metric_to_checkpoint',
        type=str,
        default='o_f1',
        choices=['o_f1', 'c_f1', 'i_f1', 'average'])

    # Flags
    parser.add_argument('--notensorboard', dest='tensorboard', action='store_false')
    parser.set_defaults(tensorboard=True)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('--nodecay_lr', dest='decay_lr', action='store_false')
    parser.set_defaults(decay_lr=True)
    parser.add_argument('--use_json_config', dest='use_json_config', action='store_true')
    parser.set_defaults(use_json_config=False)

    args = parser.parse_args()

    # If we are using configuration file, overwrite command line arguments
    if args.use_json_config:
        filename = os.path.join('../configs/', args.dataset, args.image_model + '_' + args.model_name + '.json')
        args = get_args_from_json(args, filename)

    args = set_flags_for_model(args)

    return args
