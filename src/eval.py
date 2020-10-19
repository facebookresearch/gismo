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

import torch
from torchvision import transforms

from model import get_model, label2_k_hots
from data_loader import get_loader
from utils.metrics import update_error_counts, compute_metrics
from utils.recipe1m_utils import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_loc = None if torch.cuda.is_available() else 'cpu'


def main(args):

    # Get models to test from models_path
    models_to_test = glob.glob(args.models_path + '/*.ckpt')

    # To store results
    mat_f1 = np.zeros((len(models_to_test), ))
    mat_f1_c = np.zeros((len(models_to_test), ))
    mat_f1_i = np.zeros((len(models_to_test), ))

    if not os.path.exists(args.save_results_path):
        os.makedirs(args.save_results_path)

    print('Results will be saved here: ' + args.save_results_path)

    # Extract model names
    models_to_test = [m for m in models_to_test if args.dataset in m]
    model_names = [re.split(r'[/]',m)[-1] for m in models_to_test]
   
    # Iterate over models to test
    for k, m in enumerate(models_to_test):
        print('---------------------------------------------')
        print('Evaluating ' + model_names[k])

        # Load model
        checkpoint = torch.load(m, map_location=map_loc)
        model_args = checkpoint['args']

        # Image pre-processing
        transforms_list = []
        transforms_list.append(transforms.Resize(model_args.image_size))
        transforms_list.append(transforms.CenterCrop(model_args.crop_size))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        transform = transforms.Compose(transforms_list)

        # Load data
        datapaths = json.load(open('../configs/datapaths.json'))
        dataset_root = datapaths[model_args.dataset]
        data_loader, dataset = get_loader(
            dataset=model_args.dataset,
            dataset_root=dataset_root,
            split=args.eval_split,
            transform=transform,
            batch_size=args.batch_size,
            include_eos=(model_args.decoder != 'ff'),
            shuffle=False,
            num_workers=8,
            drop_last=False,
            shuffle_labels=False)

        vocab_size = len(dataset.get_vocab())
        print('Vocabulary size is {}'.format(vocab_size))
        print('Dataset {} split contains {} images'.format(args.eval_split, len(dataset)))

        # Load model
        model = get_model(model_args, vocab_size)
        model.load_state_dict(checkpoint['state_dict'])

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

        for l, (img_inputs, target) in tqdm(enumerate(data_loader)):

            img_inputs = img_inputs.to(device)

            with torch.no_grad():
                # get model predictions
                # predictions format can either be a matrix of size batch_size x maxnumlabels, where
                # each row contains the integer labels of an image, followed by pad_value
                # or a list of sublists, where each sublist contains the integer labels of an image
                # and len(list) = batch_size and len(sublist) is variable
                _, predictions = model(
                    img_inputs, maxnumlabels=model_args.maxnumlabels, compute_predictions=True)
                # convert model predictions and targets to k-hots
                pred_k_hots = label2_k_hots(
                    predictions, vocab_size - 1, remove_eos=(model_args.decoder != 'ff'))
                target_k_hots = label2_k_hots(
                    target, vocab_size - 1, remove_eos=(model_args.decoder != 'ff'))
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

    print('Saving results...')
    data = {'Model':model_names, 'f1':mat_f1, 'f1_c':mat_f1_c, 'f1_i':mat_f1_i} 
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(args.save_results_path, 'results.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', type=str, default='voc')
        
    parser.add_argument(
        '--models_path', type=str, default='../checkpoints')
    parser.add_argument('--save_results_path', type=str, default='../checkpoints/')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--eval_split', type=str, default='test', choices=['train', 'val', 'test'])

    args = parser.parse_args()

    main(args)
