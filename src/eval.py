# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import argparse
import getpass
import glob
import re
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import hydra

from omegaconf import DictConfig, OmegaConf

import torch
from torchvision import transforms

from models.im2ingr import Im2Ingr
from models.ingredients_predictor import label2_k_hots
from loaders.recipe1m import get_loader
from utils.metrics import update_error_counts, compute_metrics
from utils.recipe1m_utils import Vocabulary
from utils.config_utils import set_flags_for_ingr_predictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    set_flags_for_ingr_predictor(cfg)

    model_names = [cfg.ingr_predictor.model]
    models_to_test = [os.path.join(cfg.checkpoint.dir, cfg.dataset.name + '_' + cfg.image_encoder.model + '_' +  cfg.ingr_predictor.model + '_' + str(cfg.misc.seed) + '.ckpt')]

    # To store results
    mat_f1 = np.zeros((len(models_to_test), ))
    mat_f1_c = np.zeros((len(models_to_test), ))
    mat_f1_i = np.zeros((len(models_to_test), ))
   
    # Iterate over models to test
    for k, m in enumerate(models_to_test):
        print('---------------------------------------------')
        print('Evaluating ' + model_names[k])

        # Load checkpoint
        checkpoint = torch.load(m, map_location=map_loc)

        # Image pre-processing
        transforms_list = []
        transforms_list.append(transforms.Resize(cfg.preprocessing.im_resize))
        transforms_list.append(transforms.CenterCrop(cfg.preprocessing.crop_size))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        transform = transforms.Compose(transforms_list)

        # Load data
        data_loader, dataset = get_loader(
            dataset=cfg.dataset.name,
            dataset_root=cfg.dataset.path,
            split='val',
            transform=transform,
            batch_size=cfg.optim.batch_size,
            include_eos=(cfg.ingr_predictor.type != 'ff'),
            shuffle=False,
            num_workers=cfg.misc.num_workers,
            drop_last=False,
            shuffle_labels=False)

        ingr_vocab_size = len(dataset.get_vocab())
        print('Vocabulary size is {}'.format(ingr_vocab_size))
        print('Dataset {} split contains {} images'.format('val', len(dataset)))

        # Build model and load model state
        model = Im2Ingr(cfg.image_encoder, cfg.ingr_predictor, ingr_vocab_size, cfg.dataset.name, cfg.dataset.maxnumlabels)
        # model.load_state_dict(checkpoint['state_dict'])

        # Eval
        model.eval()
        model = model.to(device)
        total_step = len(data_loader)
        print('Number of iterations is {}'.format(total_step))

        overall_error_counts = {
            'tp_c': 0,
            'fp_c': 0,
            'fn_c': 0,
            'tn_c': 0,
            'tp_all': 0,
            'fp_all': 0,
            'fn_all': 0
        }
        error_counts_per_card = {}
        f1s_image_per_card = {}
        card_l1_err = []
        f1s_image = []
        card_accs = []

        # for l, img_inputs, target in tqdm(enumerate(data_loader)):
        for l, (img_inputs, target) in tqdm(enumerate(data_loader)):

            img_inputs = img_inputs.to(device)

            with torch.no_grad():
                # get model predictions
                # predictions format can either be a matrix of size batch_size x maxnumlabels, where
                # each row contains the integer labels of an image, followed by pad_value
                # or a list of sublists, where each sublist contains the integer labels of an image
                # and len(list) = batch_size and len(sublist) is variable
                _, predictions = model(
                    img_inputs, maxnumlabels=cfg.dataset.maxnumlabels, compute_predictions=True)
                # convert model predictions and targets to k-hots
                pred_k_hots = label2_k_hots(
                    predictions, ingr_vocab_size - 1, remove_eos=(cfg.ingr_predictor.type != 'ff'))
                target_k_hots = label2_k_hots(
                    target, ingr_vocab_size - 1, remove_eos=(cfg.ingr_predictor.type != 'ff'))
                # update overall and per class error counts
                update_error_counts(overall_error_counts, pred_k_hots, target_k_hots)

                # get per-image error counts
                for i in range(pred_k_hots.size(0)):
                    # compute per image metrics
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

                    image_metrics = compute_metrics(image_error_counts, which_metrics=['f1'])
                    f1s_image.append(image_metrics['f1'])

        # compute overall and per class metrics
        overall_metrics = compute_metrics(overall_error_counts, ['f1', 'c_f1'], weights=None)
        overall_metrics['f1_i'] = np.mean(f1s_image)
        print(overall_metrics)

        # save results
        mat_f1[k] = overall_metrics['f1']
        mat_f1_c[k] = overall_metrics['c_f1']
        mat_f1_i[k] = overall_metrics['f1_i']


if __name__ == '__main__':
    main()
