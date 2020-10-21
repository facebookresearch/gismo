# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the 
# LICENSE file in the root directory of this source tree.

import lmdb
import numpy as np
import os
import pickle
import torch
import torch.utils.data as data

from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler
from PIL import Image


class Recipe1M(data.Dataset):

    def __init__(self,
                 data_dir,
                 split,
                 maxnumims,
                 transform=None,
                 use_lmdb=False,
                 suff='',
                 shuffle=False,
                 perm=None,
                 include_eos=False):

        self.aux_data_dir = os.path.join(data_dir, 'preprocessed')
        self.ingrs_vocab = pickle.load(
            open(os.path.join(self.aux_data_dir, suff + 'recipe1m_vocab_ingrs.pkl'), 'rb'))

        self.dataset = pickle.load(
            open(os.path.join(self.aux_data_dir, suff + 'recipe1m_' + split + '.pkl'), 'rb'))

        self.use_lmdb = use_lmdb
        if use_lmdb:
            self.image_file = lmdb.open(
                os.path.join(self.aux_data_dir, 'lmdb_' + split),
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)

        self.ids = []
        self.split = split
        for i, entry in enumerate(self.dataset):
            if len(entry['images']) == 0:
                continue
            self.ids.append(i)

        self.root = os.path.join(data_dir, 'images', split)
        self.transform = transform
        self.maxnumims = maxnumims
        self.shuffle = shuffle

        self.include_eos = include_eos
        # remove eos from vocabulary list if not needed
        if not self.include_eos:
            self.ingrs_vocab.remove_eos()

        if perm is not None:
            self.ids = np.array(self.ids)[perm]
        else:
            self.ids = np.array(self.ids)


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""

        sample = self.dataset[self.ids[index]]
        img_id = sample['id']
        paths = sample['images'][0:self.maxnumims]

        idx = index

        labels = self.dataset[self.ids[idx]]['ingredients']

        true_ingr_idxs = []
        for i in range(len(labels)):
            true_ingr_idxs.append(self.ingrs_vocab(labels[i]))
        true_ingr_idxs = list(set(true_ingr_idxs))

        if self.shuffle:
            np.random.shuffle(true_ingr_idxs)

        if self.include_eos:
            true_ingr_idxs.append(self.ingrs_vocab('<end>'))

        if len(paths) == 0:
            path = None
            image_input = torch.zeros((3, 224, 224))
        else:
            if self.split == 'train':
                img_idx = np.random.randint(0, len(paths))
            else:
                img_idx = 0
            path = paths[img_idx]
            impath = os.path.join(self.root, path[0], path[1], path[2], path[3], path)
            if self.use_lmdb:
                try:
                    with self.image_file.begin(write=False) as txn:
                        image = txn.get(path.encode())
                        image = np.fromstring(image, dtype=np.uint8)
                        image = np.reshape(image, (256, 256, 3))
                    image = Image.fromarray(image.astype('uint8'), 'RGB')
                except:
                    print("Image id not found in lmdb. Loading jpeg file...")
                    image = Image.open(impath).convert('RGB')
            else:
                image = Image.open(impath).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            image_input = image

        return image_input, true_ingr_idxs

    def __len__(self):
        return len(self.ids)

    def get_vocab(self):
        return [
            min(w, key=len) if not isinstance(w, str) else w
            for w in self.ingrs_vocab.idx2word.values()
        ]  # includes '<pad>' and eventually '<end>'


def collate_fn(data):

    img, target = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor)
    # We keep targets as list of lists (each image may contain different number of labels)
    img = torch.stack(img, 0)

    return img, target


class RandomSamplerWithStateIterator:

    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed
        g = torch.Generator()
        g.manual_seed(seed)
        self.indices = torch.randperm(len(self.data_source), generator=g).tolist()
        self.epoch = 0
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.data_source):
            raise StopIteration
        item = self.indices[self.index]
        self.index += 1
        return item

    def set_epoch(self, epoch):
        g = torch.Generator()
        g.manual_seed(self.seed + epoch)
        self.epoch = epoch
        self.indices = torch.randperm(len(self.data_source), generator=g).tolist()
        self.index = 0

    def increase_epoch(self):
        self.set_epoch(self.epoch + 1)


class RandomSamplerWithState(Sampler):

    def __init__(self, data_source, batch_size, seed):
        self.iterator = RandomSamplerWithStateIterator(data_source, seed)
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.iterator)

    def __len__(self):
        return len(self.iterator.data_source)

    def set_state(self, epoch, step):
        self.iterator.set_epoch(epoch)
        self.iterator.index = step * self.batch_size


def get_loader(dataset,
               dataset_root,
               split,
               transform,
               batch_size,
               shuffle,
               num_workers,
               include_eos,
               drop_last=False,
               shuffle_labels=False,
               seed=1234,
               checkpoint=None):

    # reads the file with ids to use for this split
    perm_file = os.path.join('../data/splits/', dataset, split + '.txt')
    with open(perm_file, 'r') as f:
        perm = np.array([int(line.rstrip('\n')) for line in f])

    if dataset == 'recipe1m':
        dataset = Recipe1M(
            dataset_root,
            split,
            maxnumims=5,
            shuffle=shuffle_labels,
            transform=transform,
            use_lmdb=False,
            suff='final_',
            perm=perm,
            include_eos=include_eos)

    def worker_init_fn(worker_id):
        np.random.seed(seed)

    if shuffle:
        # for training
        sampler = RandomSamplerWithState(dataset, batch_size, seed)
        if checkpoint is not None:
            sampler.set_state(checkpoint['args'].current_epoch, checkpoint['current_step'])
    else:
        # for validation and test
        sampler = SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        sampler=sampler)

    return data_loader, dataset


def increase_loader_epoch(data_loader):
    data_loader.sampler.iterator.increase_epoch()
