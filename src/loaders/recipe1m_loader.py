import lmdb
import numpy as np
import os
import pickle
import hydra
from PIL import Image

import torch
import torch.utils.data as data
# from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms as tf
import pytorch_lightning as pl

from loaders.recipe1m_preprocess import Vocabulary

# from utils import RandomSamplerWithState


class Recipe1M(data.Dataset):

    def __init__(self,
                 data_dir,
                 split,
                 maxnumims,
                 transform=None,
                 use_lmdb=False,
                 shuffle_labels=False,
                 split_data=None,
                 include_eos=False):

        self.aux_data_dir = os.path.join(data_dir, 'preprocessed')
        self.ingrs_vocab = pickle.load(
            open(os.path.join(self.aux_data_dir, 'final_recipe1m_vocab_ingrs.pkl'), 'rb'))

        self.dataset = pickle.load(
            open(os.path.join(self.aux_data_dir, 'final_recipe1m_' + split + '.pkl'), 'rb'))

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
        self.shuffle_labels = shuffle_labels

        self.include_eos = include_eos
        # remove eos from vocabulary list if not needed
        if not self.include_eos:
            self.ingrs_vocab.remove_eos()


        self.ids = np.array(self.ids)[split_data]

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

        if self.shuffle_labels:
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

    def get_ingr_vocab(self):
        return [
            min(w, key=len) if not isinstance(w, str) else w
            for w in self.ingrs_vocab.idx2word.values()
        ]  # includes '<pad>' and eventually '<end>'

    def get_ingr_vocab_size(self):
        return len(self.get_ingr_vocab())


class Recipe1MDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir,
                 batch_size,
                 num_workers,
                 shuffle_labels=False,
                 preprocessing=None,
                 include_eos=False,
                 seed=1234,
                 checkpoint=None):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle_labels = shuffle_labels
        self.include_eos = include_eos
        self.seed = seed
        self.num_workers = num_workers
        self.checkpoint = checkpoint  ## TODO: check how checkpoint is performed in lightning
        self.preprocessing = preprocessing
        
    def prepare_data(self):
        if not os.path.isdir(os.path.join(self.data_dir, 'preprocessed')):
            # TODO: preprocessing
            pass

    def setup(self, stage):
        if stage == 'fit':
            self.dataset_train = self._get_dataset('train')
            self.dataset_val = self._get_dataset('val')
            self.ingr_vocab_size = self.dataset_train.get_ingr_vocab_size()
            print(f'Training set composed of {len(self.dataset_train)} samples.')
            print(f'Validation set composed of {len(self.dataset_val)} samples.')
        elif stage == 'test':
            self.dataset_test = self._get_dataset('test')
            self.ingr_vocab_size = self.dataset_test.get_ingr_vocab_size()
            print(f'Test set composed of {len(self.dataset_test)} samples.')

        print(f'Ingredient vocabulary size: {self.ingr_vocab_size}.')

    def train_dataloader(self):
    
        # sampler = RandomSamplerWithState(dataset, self.batch_size, self.seed)
        # if self.checkpoint is not None:  ## TODO: review how checkpointing works here
        #     sampler.set_state(self.checkpoint['args'].current_epoch, self.checkpoint['current_step'])

        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=_collate_fn,
            worker_init_fn=self._worker_init_fn)
            # sampler=sampler)    

        return data_loader

    def val_dataloader(self):
        return self._shared_eval_dataloader('val')

    def test_dataloader(self):
        return self._shared_eval_dataloader('test')

    def _shared_eval_dataloader(self, split):

        # sampler = SequentialSampler(dataset)

        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_val if split == 'val' else self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=_collate_fn,
            worker_init_fn=self._worker_init_fn)
            # sampler=sampler)    

        return data_loader

    def _get_dataset(self, stage):

        # reads the file with ids to use for the corresponding split
        orig_cwd = hydra.utils.get_original_cwd()
        parent_orig_cwd = os.path.abspath(os.path.join(orig_cwd, os.pardir))
        splits_filename = os.path.join(parent_orig_cwd, 'data/splits/recipe1m', stage + '.txt')
            
        with open(splits_filename, 'r') as f:
            split_data = np.array([int(line.rstrip('\n')) for line in f])

        transform = self._get_transforms(stage=stage)

        dataset = Recipe1M(
            self.data_dir,
            stage,
            maxnumims=5,
            transform=transform,
            use_lmdb=True,  ## TODO: at least for recipe1m
            shuffle_labels=self.shuffle_labels,
            split_data=split_data,
            include_eos=self.include_eos)

        return dataset

    def _get_transforms(self, stage):
        transforms_list = [tf.Resize(self.preprocessing.im_resize)]

        if stage == 'train':
            transforms_list.append(tf.RandomHorizontalFlip())
            transforms_list.append(tf.RandomAffine(degrees=10, translate=(0.1, 0.1)))
            transforms_list.append(tf.RandomCrop(self.preprocessing.crop_size))   
        else:
            transforms_list.append(tf.CenterCrop(self.preprocessing.crop_size))
            
        transforms_list.append(tf.ToTensor())
        transforms_list.append(tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        transforms = tf.Compose(transforms_list)

        return transforms

    def _worker_init_fn(self, worker_id):
        np.random.seed(self.seed)

def _collate_fn(data):

    img, target = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor)
    # We keep targets as list of lists (each image may contain different number of labels)
    img = torch.stack(img, 0)

    return img, target


