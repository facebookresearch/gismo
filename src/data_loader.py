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
from utils.voc_utils import VOCDetection, ET
from PIL import Image

category_map = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}


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


class VOC(VOCDetection):

    def __init__(self,
                 root='',
                 year='2007',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 shuffle=True,
                 perm=None,
                 include_eos=False):

        if image_set in ['train', 'val']:
            set = 'trainval'
        else:
            set = 'test'

        VOCDetection.__init__(
            self,
            root=root,
            year=year,
            image_set=set,
            download=download,
            transform=transform,
            target_transform=target_transform)

        self.cats = [
            'eos', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor', '<pad>'
        ]

        if perm is not None:
            self.images = np.array(self.images)[perm]
            self.annotations = np.array(self.annotations)[perm]
        else:
            self.images = np.array(self.images)
            self.annotations = np.array(self.annotations)

        self.include_eos = include_eos

        self.shuffle = shuffle

        # remove eos from category list if not needed
        if not self.include_eos:
            self.cats = self.cats[1:]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """

        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        if self.transform is not None:
            img = self.transform(img)

        target = target['annotation']['object']
        # handle 1-object case
        if type(target) is dict:
            target = [target]

        idxs = list(range(len(target)))
        if self.shuffle:
            np.random.shuffle(idxs)

        # build target
        target_list = []
        for t in idxs:
            category_name = target[t]['name']
            category_id = self.cats.index(category_name)
            if category_id not in target_list:
                target_list.append(category_id)

        # eos
        if self.include_eos:
            target_list.append(0)

        return img, target_list

    def get_vocab(self):
        return self.cats

    def get_sample_list(self):
        return self.images


class NUSWIDE(data.Dataset):

    def __init__(self,
                 root,
                 split,
                 transform=None,
                 shuffle=False,
                 perm=None,
                 include_eos=False):

        self.root = root
        self.transform = transform
        self.shuffle = shuffle
        self.include_eos = include_eos

        labels_dir = os.path.join(self.root, 'Concepts81.txt')

        lines = list(open(labels_dir, 'r'))

        self.category_list = ['eos'] + lines + ['<pad>']

        # remove eos from category list if not needed
        if not self.include_eos:
            self.category_list = self.category_list[1:]

        if split == 'train' or split == 'val':
            self.tags = self.load_tags('train')
            self.ids = list(open(os.path.join(self.root, 'ImageList', 'TrainImagelist.txt')))
        else:
            self.tags = self.load_tags('test')
            self.ids = list(open(os.path.join(self.root, 'ImageList', 'TestImagelist.txt')))

        if perm is not None:
            self.tags = np.array(self.tags)[perm]
            self.ids = np.array(self.ids)[perm]
        else:
            self.tags = np.array(self.tags)
            self.ids = np.array(self.ids)

        self.category_list = [x.rstrip() for x in self.category_list]


    def load_tags(self, split):

        anns = []
        for cat in self.category_list[1 if self.include_eos else 0:-1]:

            c_anns = list(
                open(
                    os.path.join(self.root, 'TrainTestLabels',
                                 'Labels_' + cat.rstrip() + '_' + split.capitalize() + '.txt')))

            if len(anns) == 0:
                anns = c_anns
            else:
                for i in range(len(c_anns)):
                    anns[i] += ' ' + c_anns[i]
        return anns

    def __getitem__(self, item):

        path = self.ids[item].rstrip()
        data = self.tags[item]

        path = '/'.join(path.split('\\'))
        impath = os.path.join(self.root, 'Flickr', path)
        img = Image.open(impath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # data = data.rstrip().split(' ')
        # data = np.array([int(i.rstrip()) for i in data])
        data = np.asarray(data.rstrip().split('\n '), dtype='uint8')
        target = np.where(data == 1)[0]

        idxs = list(range(len(target)))
        if self.shuffle:
            np.random.shuffle(idxs)

        # build target
        target_list = []
        for t in idxs:
            category_id = target[t] + 1 if self.include_eos else target[t]
            if category_id not in target_list:
                target_list.append(category_id)

        # eos
        if self.include_eos:
            target_list.append(0)

        return img, target_list

    def __len__(self):
        return len(self.ids)

    def get_vocab(self):
        return self.category_list

    def get_sample_list(self):
        return self.ids


class COCO(data.Dataset):

    def __init__(self,
                 root,
                 annFile,
                 transform=None,
                 shuffle=False,
                 perm=None,
                 include_eos=False,):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())

        if perm is not None:
            self.ids = np.array(self.ids)[perm].tolist()

        self.transform = transform
        self.shuffle = shuffle

        self.category_list = [
            'eos', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

        self.category_list.append('<pad>')

        self.include_eos = include_eos

        # remove eos from category list if not needed
        if not self.include_eos:
            self.category_list = self.category_list[1:]

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        impath = os.path.join(self.root, path)
        img = Image.open(impath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        idxs = list(range(len(target)))
        if self.shuffle:
            np.random.shuffle(idxs)

        # build target
        target_list = []
        for t in idxs:
            category_id = target[t]['category_id']
            category_name = category_map[category_id]
            category_id = self.category_list.index(category_name)
            if category_id not in target_list:
                target_list.append(category_id)

        # eos
        if self.include_eos:
            target_list.append(0)

        return img, target_list

    def __len__(self):
        return len(self.ids)

    def get_vocab(self):
        return self.category_list

    def get_sample_list(self):
        return self.ids


class ADE20K(data.Dataset):

    def __init__(self,
                 root,
                 split,
                 transform=None,
                 shuffle=False,
                 perm=None,
                 include_eos=False):

        labels_dir = os.path.join(root, 'objectInfo150.txt')
        lines = [x.split('\t')[-1].split(',')[0].rstrip() for x in open(labels_dir).readlines()][1:]

        self.category_list = ['eos'] + lines + ['<pad>']

        self.root = root
        if split == 'train' or split == 'val':
            samples_dir = os.path.join(root, 'ADE20K_object150_train.txt')
            self.ids = list(open(samples_dir, 'r'))
        else:
            samples_dir = os.path.join(root, 'ADE20K_object150_val.txt')
            self.ids = list(open(samples_dir, 'r'))

        if perm is not None:
            self.ids = np.array(self.ids)[perm]
        else:
            self.ids = np.array(self.ids)

        self.transform = transform
        self.shuffle = shuffle

        self.include_eos = include_eos

        # remove eos from category list if not needed
        if not self.include_eos:
            self.category_list = self.category_list[1:]

    def __getitem__(self, item):

        sample_name = self.ids[item].rstrip()
        impath = os.path.join(self.root, 'images', sample_name)

        maskname = sample_name.split('.jpg')[0] + '.png'
        maskpath = os.path.join(self.root, 'annotations', maskname)

        img = Image.open(impath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        mask = Image.open(maskpath)

        idxs = np.unique(mask)
        # the 0 idx refers to "other objects"
        idxs = np.delete(idxs, np.where(idxs == 0))

        if not self.include_eos:
            idxs = [i-1 for i in idxs]

        if self.shuffle:
            np.random.shuffle(idxs)

        target_list = []
        for t in idxs:
            category_id = t
            if category_id not in target_list:
                target_list.append(category_id)

        # eos
        if self.include_eos:
            target_list.append(0)

        return img, target_list

    def __len__(self):
        return len(self.ids)

    def get_vocab(self):
        return self.category_list

    def get_sample_list(self):
        return self.ids


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

    if dataset == 'coco':
        if split == 'train' or split == 'val':
            annFile = os.path.join(dataset_root, 'annotations', 'instances_train2014.json')
            impath = os.path.join(dataset_root, 'train2014')
        else:
            annFile = os.path.join(dataset_root, 'annotations', 'instances_val2014.json')
            impath = os.path.join(dataset_root, 'val2014')

        dataset = COCO(
            root=impath,
            annFile=annFile,
            transform=transform,
            shuffle=shuffle_labels,
            perm=perm,
            include_eos=include_eos)

    elif dataset == 'voc':
        dataset = VOC(
            root=dataset_root,
            year='2007',
            image_set=split,
            download=False,
            transform=transform,
            shuffle=shuffle_labels,
            perm=perm,
            include_eos=include_eos)

    elif dataset == 'nuswide':
        dataset = NUSWIDE(
            dataset_root,
            split,
            transform=transform,
            shuffle=shuffle_labels,
            perm=perm,
            include_eos=include_eos)

    elif dataset == 'ade20k':
        dataset = ADE20K(
            dataset_root,
            split,
            transform=transform,
            shuffle=shuffle_labels,
            perm=perm,
            include_eos=include_eos)

    elif dataset == 'recipe1m':
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
