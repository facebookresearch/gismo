# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
# Code adapted from https://github.com/facebookresearch/inversecooking
# This source code is licensed under the MIT license found in the
# LICENSE file in https://github.com/facebookresearch/inversecooking

import glob
import os
from operator import itemgetter

import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, checkpoints_dir, name, resume=False):
        self.win_size = 256
        self.name = name
        self.saved = False
        self.checkpoints_dir = checkpoints_dir
        self.ncols = 4

        # remove existing
        if not resume:
            for filename in glob.glob(self.checkpoints_dir + "/events*"):
                os.remove(filename)
        self.writer = SummaryWriter(checkpoints_dir)

    def reset(self):
        self.saved = False

    # images: (b, c, 0, 1) array of images
    def image_summary(self, mode, epoch, images):
        images = vutils.make_grid(images, normalize=True, scale_each=True)
        self.writer.add_image("{}/Image".format(mode), images, epoch)

    # text: type: ingredients/recipe
    def text_summary(self, mode, epoch, type, text, vocabulary, gt=True, max_length=20):
        for i, el in enumerate(text):  # text_list
            if not gt:  # we are printing a sample
                idx = el.nonzero().squeeze() + 1
            else:
                idx = el  # we are printing the ground truth

            words_list = itemgetter(*idx)(vocabulary)

            if len(words_list) <= max_length:
                self.writer.add_text(
                    "{}/{}_{}_{}".format(mode, type, i, "gt" if gt else "prediction"),
                    ", ".join(filter(lambda x: x != "<pad>", words_list)),
                    epoch,
                )
            else:
                self.writer.add_text(
                    "{}/{}_{}_{}".format(mode, type, i, "gt" if gt else "prediction"),
                    "Number of sampled ingredients is too big: {}".format(
                        len(words_list)
                    ),
                    epoch,
                )

    # losses: dictionary of error labels and values
    def scalar_summary(self, mode, epoch, **args):
        for k, v in args.items():
            self.writer.add_scalar("{}/{}".format(mode, k), v, epoch)

        self.writer.export_scalars_to_json(
            "{}/tensorboard_all_scalars.json".format(self.checkpoints_dir)
        )

    def histo_summary(self, model, step):
        """Log a histogram of the tensor of values."""

        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, step)

    def close(self):
        self.writer.close()
