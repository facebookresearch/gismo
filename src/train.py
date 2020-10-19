# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import random
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms

from data_loader import get_loader, increase_loader_epoch
from model import get_model

from args import get_parser
from model import label2_k_hots
from utils.tb_visualizer import Visualizer
from utils.metrics import update_error_counts, compute_metrics
from utils.recipe1m_utils import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

MAIN_PID = os.getpid()
CHECKPOINT_tempfile = ''
HALT_filename = ''


class StreamToLogger(object):
    """
   Fake file-like stream object that redirects writes to a logger instance.
   """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def save_checkpoint(model, optimizer, args, es_best, epoch_best, current_step, current_pat,
                    checkpoint_filename):
    checkpoint = {}
    if torch.cuda.device_count() > 1:
        checkpoint['state_dict'] = model.module.state_dict()
    else:
        checkpoint['state_dict'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['es_best'] = es_best
    checkpoint['epoch_best'] = epoch_best
    checkpoint['args'] = args
    checkpoint['current_step'] = current_step
    checkpoint['current_pat'] = current_pat

    print('saving tmp checkpoint', CHECKPOINT_tempfile, flush=True)
    torch.save(checkpoint, CHECKPOINT_tempfile)
    if os.path.isfile(CHECKPOINT_tempfile):
        print('file was saved correctly. renaming now', flush=True)
        os.rename(CHECKPOINT_tempfile, checkpoint_filename + '.ckpt')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def main(args):
    global HALT_filename, CHECKPOINT_tempfile

    # Create model directory & other aux folders for logging
    where_to_save = os.path.join(args.save_dir, args.dataset, args.model_name, args.image_model, args.experiment_name)
    checkpoints_dir = os.path.join(where_to_save, 'checkpoints')
    suffix = '_'.join([args.dataset, args.model_name, str(args.seed)])
    checkpoint_filename = os.path.join(checkpoints_dir, '_'.join([suffix, 'checkpoint']))
    print(checkpoint_filename)
    logs_dir = os.path.join(where_to_save, 'logs')
    tb_logs = os.path.join(where_to_save, 'tb_logs', args.dataset,
                           args.model_name + '_' + str(args.seed))
    make_dir(where_to_save)
    make_dir(logs_dir)
    make_dir(checkpoints_dir)
    make_dir(tb_logs)

    # Create loggers
    # stdout logger
    stdout_logger = logging.getLogger('STDOUT')
    stdout_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s: %(message)s')
    fh_out = logging.FileHandler(os.path.join(logs_dir, 'train_{}.log'.format(suffix)))
    fh_out.setFormatter(formatter)
    stdout_logger.addHandler(fh_out)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    stdout_logger.addHandler(ch)
    # stderr logger
    stderr_logger = logging.getLogger('STDERR')
    fh_err = logging.FileHandler(os.path.join(logs_dir, 'train_{}.err'.format(suffix)), mode='w')
    fh_err.setFormatter(formatter)
    stderr_logger.addHandler(fh_err)
    sl_stderr = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl_stderr

    # HALT file is used as a sign of job completion.
    # Check if no HALT file left from previous runs.
    HALT_filename = os.path.join(where_to_save, 'HALT_{}'.format(suffix))
    if os.path.isfile(HALT_filename):
        os.remove(HALT_filename)

    # Remove CHECKPOINT_tempfile
    CHECKPOINT_tempfile = checkpoint_filename + '.tmp.ckpt'
    if os.path.isfile(CHECKPOINT_tempfile):
        os.remove(CHECKPOINT_tempfile)

    # Create tensorboard visualizer
    if args.tensorboard:
        logger = Visualizer(tb_logs, name='visual_results', resume=args.resume)

    # Check if we want to resume from last checkpoint of current model
    checkpoint = None
    if args.resume:
        if os.path.isfile(checkpoint_filename + '.ckpt'):
            checkpoint = torch.load(checkpoint_filename + '.ckpt', map_location=map_loc)
            num_epochs = args.num_epochs
            args = checkpoint['args']
            args.num_epochs = num_epochs

    # Build data loader
    data_loaders = {}
    datasets = {}
    for split in ['train', 'val']:

        transforms_list = [transforms.Resize(args.image_size)]

        # Image pre-processing
        if split == 'train':
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
            transforms_list.append(transforms.RandomCrop(args.crop_size))

        else:
            transforms_list.append(transforms.CenterCrop(args.crop_size))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        transform = transforms.Compose(transforms_list)

        # Load dataset path
        datapaths = json.load(open('../configs/datapaths.json'))
        dataset_root = datapaths[args.dataset]
        data_loaders[split], datasets[split] = get_loader(
            dataset=args.dataset,
            dataset_root=dataset_root,
            split=split,
            transform=transform,
            batch_size=args.batch_size,
            include_eos=(args.decoder != 'ff'),
            shuffle=(split == 'train'),
            num_workers=args.num_workers,
            drop_last=(split == 'train'),
            shuffle_labels=args.shuffle_labels,
            seed=args.seed,
            checkpoint=checkpoint)
        stdout_logger.info('Dataset {} split contains {} images'.format(
            split, len(datasets[split])))

    vocab_size = len(datasets[split].get_vocab())
    stdout_logger.info('Vocabulary size is {}'.format(vocab_size))

    # Build the model
    model = get_model(args, vocab_size)

    # add model parameters
    if model.image_encoder.last_module is not None:
        params = list(model.decoder.parameters()) + list(
            model.image_encoder.last_module.parameters())
    else:
        params = list(model.decoder.parameters())
    params_cnn = list(model.image_encoder.pretrained_net.parameters())

    n_p_cnn = sum(p.numel() for p in params_cnn if p.requires_grad)
    n_p = sum(p.numel() for p in params if p.requires_grad)
    total = n_p + n_p_cnn
    stdout_logger.info("CNN params: {}".format(n_p_cnn))
    stdout_logger.info("decoder params: {}".format(n_p))
    stdout_logger.info("total params: {}".format(total))

    # encoder and decoder optimizers
    if params_cnn is not None and args.finetune_after == 0:
        optimizer = torch.optim.Adam(
            [{
                'params': params
            }, {
                'params': params_cnn,
                'lr': args.learning_rate * args.scale_learning_rate_cnn
            }],
            lr=args.learning_rate,
            weight_decay=args.weight_decay)
        keep_cnn_gradients = True
        stdout_logger.info("Fine tuning image encoder")
    else:
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)
        keep_cnn_gradients = False
        stdout_logger.info("Freezing image encoder")

    # early stopping and checkpoint
    es_best = {'o_f1': 0, 'c_f1': 0, 'i_f1': 0, 'average': 0}
    epoch_best = {'o_f1': -1, 'c_f1': -1, 'i_f1': -1, 'average': -1}
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        es_best = checkpoint['es_best']
        epoch_best = checkpoint['epoch_best']

    if device != 'cpu' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    cudnn.benchmark = True

    if not hasattr(args, 'current_epoch'):
        args.current_epoch = 0

    # Train the model
    decay_factor = 1.0
    start_step = 0 if checkpoint is None else checkpoint['current_step']
    curr_pat = 0 if checkpoint is None else checkpoint['current_pat']
    for epoch in range(args.current_epoch, args.num_epochs):

        # save current epoch for resuming
        if args.tensorboard:
            logger.reset()

        # increase / decrease values for moving params
        if args.decay_lr:
            frac = epoch // args.lr_decay_every
            decay_factor = args.lr_decay_rate**frac
            new_lr = args.learning_rate * decay_factor
            stdout_logger.info('Epoch %d. lr: %.5f' % (epoch, new_lr))
            set_lr(optimizer, decay_factor)

        if args.finetune_after != -1 and args.finetune_after < epoch \
                and not keep_cnn_gradients and params_cnn is not None:

            stdout_logger.info("Starting to fine tune CNN")
            # start with learning rates as they were (if decayed during training)
            optimizer = torch.optim.Adam(
                [{
                    'params': params
                }, {
                    'params': params_cnn,
                    'lr': decay_factor * args.learning_rate * args.scale_learning_rate_cnn
                }],
                lr=decay_factor * args.learning_rate)
            keep_cnn_gradients = True

        for split in ['train', 'val']:

            if split == 'train':
                model.train()
            else:
                model.eval()
            total_step = len(data_loaders[split])
            loader = iter(data_loaders[split])

            total_loss_dict = {
                'label_loss': [],
                'eos_loss': [],
                'cardinality_loss': [],
                'loss': [],
                'o_f1': [],
                'c_f1': [],
                'i_f1': [],
            }

            torch.cuda.synchronize()
            start = time.time()

            overall_error_counts = {
                'tp_c': 0,
                'fp_c': 0,
                'fn_c': 0,
                'tn_c': 0,
                'tp_all': 0,
                'fp_all': 0,
                'fn_all': 0
            }

            i = 0 if split == 'val' else start_step
            for info in loader:
                img_inputs, gt = info

                # adapt gts by adding pad_value to match maxnumlabel length
                gt = [
                    sublist + [vocab_size - 1] * (args.maxnumlabels - len(sublist))
                    for sublist in gt
                ]
                gt = torch.LongTensor(gt)

                # move to device
                img_inputs = img_inputs.to(device)
                gt = gt.to(device)

                loss_dict = {}

                if split == 'val':
                    with torch.no_grad():
                        # get losses and label predictions
                        _, predictions = model(
                            img_inputs,
                            maxnumlabels=args.maxnumlabels,
                            compute_losses=False,
                            compute_predictions=True)

                        # convert model predictions and targets to k-hots
                        pred_k_hots = label2_k_hots(
                            predictions, vocab_size - 1, remove_eos=(args.decoder != 'ff'))
                        target_k_hots = label2_k_hots(
                            gt, vocab_size - 1, remove_eos=(args.decoder != 'ff'))

                        # update overall and per class error types
                        update_error_counts(overall_error_counts, pred_k_hots, target_k_hots)

                        # update per image error types
                        i_f1s = []
                        for i in range(pred_k_hots.size(0)):
                            image_error_counts = {
                                'tp_c': 0,
                                'fp_c': 0,
                                'fn_c': 0,
                                'tn_c': 0,
                                'tp_all': 0,
                                'fp_all': 0,
                                'fn_all': 0
                            }
                            update_error_counts(image_error_counts, pred_k_hots[i].unsqueeze(0),
                                                target_k_hots[i].unsqueeze(0))
                            image_metrics = compute_metrics(
                                image_error_counts, which_metrics=['f1'])
                            i_f1s.append(image_metrics['f1'])

                        loss_dict['i_f1'] = np.mean(i_f1s)
                        del predictions, pred_k_hots, target_k_hots, image_metrics

                else:
                    losses, _ = model(
                        img_inputs,
                        gt,
                        maxnumlabels=args.maxnumlabels,
                        keep_cnn_gradients=keep_cnn_gradients,
                        compute_losses=True)

                    # label loss
                    label_loss = losses['label_loss']
                    label_loss = label_loss.mean()
                    loss_dict['label_loss'] = label_loss.item()

                    # cardinality loss
                    if args.pred_cardinality != 'none':
                        cardinality_loss = losses['cardinality_loss']
                        cardinality_loss = cardinality_loss.mean()
                        loss_dict['cardinality_loss'] = cardinality_loss.item()
                    else:
                        cardinality_loss = 0

                    # eos loss
                    if args.perminv:
                        eos_loss = losses['eos_loss']
                        eos_loss = eos_loss.mean()
                        loss_dict['eos_loss'] = eos_loss.item()
                    else:
                        eos_loss = 0

                    # total loss
                    loss = args.loss_weight[0] * label_loss \
                           + args.loss_weight[1]*cardinality_loss + \
                           args.loss_weight[2]*eos_loss
                    loss_dict['loss'] = loss.item()

                    # optimizer step
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    del loss, losses
                del img_inputs

                for key in loss_dict.keys():
                    total_loss_dict[key].append(loss_dict[key])

                # Print log info
                if args.log_step != -1 and i % args.log_step == 0:
                    elapsed_time = time.time() - start
                    lossesstr = ""
                    for k in total_loss_dict.keys():
                        if len(total_loss_dict[k]) == 0:
                            continue
                        this_one = "%s: %.4f" % (k, np.mean(total_loss_dict[k][-args.log_step:]))
                        lossesstr += this_one + ', '
                    # this only displays nll loss on captions, the rest of losses will
                    # be in tensorboard logs
                    strtoprint = 'Split: %s, Epoch [%d/%d], Step [%d/%d], Losses: %sTime: %.4f' % (
                        split, epoch, args.num_epochs, i, total_step, lossesstr, elapsed_time)
                    stdout_logger.info(strtoprint)
                    if args.tensorboard and split == 'train':
                        logger.scalar_summary(
                            mode=split + '_iter',
                            epoch=total_step * epoch + i,
                            **{
                                k: np.mean(v[-args.log_step:])
                                for k, v in total_loss_dict.items()
                                if v
                            })

                    torch.cuda.synchronize()
                    start = time.time()

                i += 1

            if split == 'train':
                increase_loader_epoch(data_loaders['train'])
                start_step = 0

            if split == 'val':
                overal_metrics = compute_metrics(overall_error_counts, ['f1', 'c_f1'], weights=None)

                total_loss_dict['o_f1'] = overal_metrics['f1']
                total_loss_dict['c_f1'] = overal_metrics['c_f1']

                if args.tensorboard:
                    # 1. Log scalar values (scalar summary)
                    logger.scalar_summary(
                        mode=split,
                        epoch=epoch,
                        **{k: np.mean(v)
                           for k, v in total_loss_dict.items()
                           if v})

        # early stopping
        metric_average = 0
        best_at_checkpoint_metric = False
        if args.metric_to_checkpoint != 'average':
            es_value = np.mean(total_loss_dict[args.metric_to_checkpoint])
            if es_value > es_best[args.metric_to_checkpoint]:
                es_best[args.metric_to_checkpoint] = es_value
                epoch_best[args.metric_to_checkpoint] = epoch
                best_at_checkpoint_metric = True
                save_checkpoint(model, optimizer, args, es_best, epoch_best, 0, 0,
                                '{}.best.{}'.format(checkpoint_filename, args.metric_to_checkpoint))
        else:
            for metric in ['o_f1', 'c_f1', 'i_f1']:
                es_value = np.mean(total_loss_dict[metric])
                metric_average += es_value
            metric_average /= 3
            if metric_average > es_best['average']:
                es_best['average'] = metric_average
                epoch_best['average'] = epoch
                if 'average' == args.metric_to_checkpoint:
                    best_at_checkpoint_metric = True
                    save_checkpoint(model, optimizer, args, es_best, epoch_best, 0, 0,
                                    '{}.best.average'.format(checkpoint_filename))

        if best_at_checkpoint_metric:
            curr_pat = 0
        else:
            curr_pat += 1

        args.current_epoch = epoch + 1  # Save the epoch at which the model needs to start
        save_checkpoint(model, optimizer, args, es_best, epoch_best, 0, curr_pat,
                        checkpoint_filename)
        stdout_logger.info('Saved checkpoint for epoch {}.'.format(epoch))

        if curr_pat > args.patience:
            break

    # Mark job as finished
    f = open(HALT_filename, 'w')
    for metric in es_best.keys():
        f.write('{}:{}\n'.format(metric, es_best[metric]))
    f.close()

    if args.tensorboard:
        logger.close()


if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    main(args)
