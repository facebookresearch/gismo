import os
import pickle
from typing import Tuple

import lmdb
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class Recipe1M(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        maxnumims: int,
        maxnumlabels: int,
        maxnuminstrs: int = 10,
        maxinstrlength: int = 15,
        return_img=False,
        return_ingr=False,
        return_recipe=False,
        transform=None,
        use_lmdb=False,
        shuffle_labels=False,
        split_data=None,
        include_eos=False,
    ):
        self.root = os.path.join(data_dir, "images", split)
        self.aux_data_dir = os.path.join(data_dir, "preprocessed")
        self.split = split
        self.maxnumims = maxnumims
        self.maxnumlabels = maxnumlabels
        self.maxseqlen = maxnuminstrs * maxinstrlength
        self.return_img = return_img
        self.return_ingr = return_ingr
        self.return_recipe = return_recipe
        self.transform = transform
        self.use_lmdb = use_lmdb
        self.shuffle_labels = shuffle_labels
        self.include_eos = include_eos

        self.ingr_vocab = []
        self.instr_vocab = []
        self.dataset = []

        # load ingredient voc
        if self.return_ingr:
            self.ingr_vocab = pickle.load(
                open(
                    os.path.join(self.aux_data_dir, "final_recipe1m_vocab_ingrs.pkl"),
                    "rb",
                )
            )

            # remove eos from vocabulary list if not needed
            if not self.include_eos:
                self.ingr_vocab.remove_eos()
                self.ingr_pad_value = self.get_ingr_vocab_size() - 1
                self.ingr_eos_value = self.ingr_pad_value
            else:
                self.ingr_pad_value = self.get_ingr_vocab_size() - 1
                self.ingr_eos_value = self.ingr_vocab("<end>")

        # load recipe instructions voc
        if self.return_recipe:
            self.instr_vocab = pickle.load(
                open(
                    os.path.join(self.aux_data_dir, "final_recipe1m_vocab_toks.pkl"),
                    "rb",
                )
            )

        # load dataset
        if self.return_img or self.return_ingr or self.return_recipe:
            self.dataset = pickle.load(
                open(
                    os.path.join(self.aux_data_dir, "final_recipe1m_" + split + ".pkl"),
                    "rb",
                )
            )
        else:
            raise ValueError(
                """Dataset loader asked to not return images, nor ingredients, nor recipes. 
                                Please set either return_images, return_ingr or return_recipe to True."""
            )

        if use_lmdb:
            # open lmdb file
            self.image_file = lmdb.open(
                os.path.join(self.aux_data_dir, "lmdb_" + split),
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        # get ids of data samples used
        ids = []
        for i, entry in enumerate(self.dataset):
            if len(entry["images"]) == 0:
                continue
            ids.append(i)
        ids = np.array(ids)[split_data]

        # prune dataset
        self.dataset = [self.dataset[i] for i in ids]

    def __getitem__(self, index: int) -> Tuple[Image.Image, "ingredients", "recipe"]:
        ret_img = None
        ret_ingr = None
        ret_rec = None

        # get dataset sample
        sample = self.dataset[index]

        # get set of ingredients
        if self.return_ingr:
            ingr = self.dataset[index]["ingredients"]

            # ingredients to idx
            true_ingr_idxs = []
            for i in range(len(ingr)):
                true_ingr_idxs.append(self.ingr_vocab(ingr[i]))
            true_ingr_idxs = list(set(true_ingr_idxs))

            if self.shuffle_labels:
                np.random.shuffle(true_ingr_idxs)

            if self.include_eos:
                true_ingr_idxs.append(self.ingr_vocab("<end>"))

            ret_ingr = true_ingr_idxs + [self.ingr_pad_value] * (
                self.maxnumlabels + self.include_eos - len(true_ingr_idxs)
            )

        # get image
        if self.return_img:
            sample["id"]
            paths = sample["images"][0 : self.maxnumims]

            # images
            if len(paths) == 0:
                path = None
                ret_img = torch.zeros((3, 224, 224))  # TODO: ???
            else:
                if self.split == "train":
                    img_idx = np.random.randint(0, len(paths))
                else:
                    img_idx = 0
                path = paths[img_idx]
                impath = os.path.join(
                    self.root, path[0], path[1], path[2], path[3], path
                )

                if self.use_lmdb:
                    try:
                        with self.image_file.begin(write=False) as txn:
                            image = txn.get(path.encode())
                            image = np.frombuffer(image, dtype=np.uint8)
                            image = np.reshape(image, (256, 256, 3))
                        image = Image.fromarray(image.astype("uint8"), "RGB")
                    except:
                        print("Image id not found in lmdb. Loading jpeg file...")
                        image = Image.open(impath).convert("RGB")
                else:
                    image = Image.open(impath).convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)
                ret_img = image

        # get recipe (title and instructions)
        if self.return_recipe:
            title = sample["title"]
            instructions = sample["tokenized"]

            tokens = []
            tokens.extend(title)
            # add fake token to separate title from recipe
            tokens.append("<eoi>")
            for i in instructions:
                tokens.extend(i)
                tokens.append("<eoi>")

            # Convert recipe (string) to word ids.
            ret_rec = []
            ret_rec = self.recipe_to_idxs(tokens, ret_rec)
            ret_rec.append(self.instr_vocab("<end>"))

            ret_rec = ret_rec[0 : self.maxseqlen]
            ret_rec = ret_rec + [self.instr_vocab("<pad>")] * (
                self.maxseqlen - len(ret_rec)
            )

        return ret_img, ret_ingr, ret_rec

    def __len__(self):
        return len(self.dataset)

    def get_ingr_vocab(self):
        return [
            min(w, key=len) if not isinstance(w, str) else w
            for w in self.ingr_vocab.idx2word.values()
        ]  # includes '<pad>' and eventually '<end>'

    def get_ingr_vocab_size(self):
        return len(self.get_ingr_vocab())

    def get_instr_vocab(self):
        return self.instr_vocab

    def get_instr_vocab_size(self):
        return len(self.get_instr_vocab())

    def recipe_to_idxs(self, tokens, recipe):
        recipe.append(self.instr_vocab("<start>"))
        for token in tokens:
            recipe.append(self.instr_vocab(token))
        return recipe


class Recipe1MDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        splits_path: str,
        maxnumlabels: int,
        batch_size: int,
        num_workers: int,
        maxnuminstrs: int = 10,
        maxinstrlength: int = 15,
        shuffle_labels=False,
        preprocessing=None,
        include_eos=False,
        seed=1234,
        checkpoint=None,
        return_img=False,
        return_ingr=False,
        return_recipe=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.splits_path = splits_path
        self.maxnumlabels = maxnumlabels
        self.maxnuminstrs = maxnuminstrs
        self.maxinstrlength = maxinstrlength
        self.batch_size = batch_size
        self.shuffle_labels = shuffle_labels
        self.include_eos = include_eos
        self.seed = seed
        self.num_workers = num_workers
        self.checkpoint = (
            checkpoint  ## TODO: check how checkpoint is performed in lightning
        )
        self.preprocessing = preprocessing
        self.return_img = return_img
        self.return_ingr = return_ingr
        self.return_recipe = return_recipe

    def prepare_data(self):
        if not os.path.isdir(os.path.join(self.data_dir, "preprocessed")):
            # TODO: preprocessing
            pass

    def setup(self, stage: str):
        if stage == "fit":
            self.dataset_train = self._get_dataset("train")
            self.dataset_val = self._get_dataset("val")
            self.ingr_vocab_size = self.dataset_train.get_ingr_vocab_size()
            self.instr_vocab_size = self.dataset_train.get_instr_vocab_size()
            self.ingr_eos_value = self.dataset_train.ingr_eos_value
            print(f"Training set composed of {len(self.dataset_train)} samples.")
            print(f"Validation set composed of {len(self.dataset_val)} samples.")
        elif stage == "test":
            self.dataset_test = self._get_dataset("test")
            print(f"Test set composed of {len(self.dataset_test)} samples.")
            self.ingr_vocab_size = self.dataset_test.get_ingr_vocab_size()
            self.instr_vocab_size = self.dataset_test.get_instr_vocab_size()
            self.ingr_eos_value = self.dataset_test.ingr_eos_value

        print(f"Ingredient vocabulary size: {self.ingr_vocab_size}.")
        print(f"Instruction vocabulary size: {self.instr_vocab_size}.")

    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=_collate_fn,
            worker_init_fn=self._worker_init_fn,
        )
        return data_loader

    def val_dataloader(self):
        return self._shared_eval_dataloader("val")

    def test_dataloader(self):
        return self._shared_eval_dataloader("test")

    def _shared_eval_dataloader(self, split: str):
        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_val if split == "val" else self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=_collate_fn,
            worker_init_fn=self._worker_init_fn,
        )
        return data_loader

    def _get_dataset(self, stage: str):

        # reads the file with ids to use for the corresponding split
        splits_filename = os.path.join(self.splits_path, stage + ".txt")
        with open(splits_filename, "r") as f:
            split_data = np.array([int(line.rstrip("\n")) for line in f])

        dataset = Recipe1M(
            self.data_dir,
            stage,
            maxnumims=5,
            maxnumlabels=self.maxnumlabels,
            maxnuminstrs=self.maxnuminstrs,
            maxinstrlength=self.maxinstrlength,
            transform=self._get_transforms(stage=stage),
            use_lmdb=False,  # TODO - check if necessary
            shuffle_labels=self.shuffle_labels,
            split_data=split_data,
            include_eos=self.include_eos,
            return_img=self.return_img,
            return_ingr=self.return_ingr,
            return_recipe=self.return_recipe,
        )
        return dataset

    def _get_transforms(self, stage):
        pipeline = [transforms.Resize(self.preprocessing.im_resize)]
        if stage == "train":
            pipeline.append(transforms.RandomHorizontalFlip())
            pipeline.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
            pipeline.append(transforms.RandomCrop(self.preprocessing.crop_size))
        else:
            pipeline.append(transforms.CenterCrop(self.preprocessing.crop_size))
        pipeline.append(transforms.ToTensor())
        pipeline.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        return transforms.Compose(pipeline)

    def _worker_init_fn(self, worker_id):
        np.random.seed(self.seed)


def _collate_fn(data):

    img, ingr, recipe = zip(*data)

    ret = {}

    # Merge images, ingredients and recipes in minibatch
    if img[0] is not None:
        ret["img"] = torch.stack(img, 0)

    if ingr[0] is not None:
        ret["ingr_gt"] = torch.tensor(ingr)

    if recipe[0] is not None:
        ret["recipe_gt"] = torch.tensor(recipe)

    return ret


# if __name__ == '__main__':

#     splits_filename = os.path.join('../data/splits/recipe1m', 'train' + '.txt')

#     with open(splits_filename, 'r') as f:
#         split_data = np.array([int(line.rstrip('\n')) for line in f])

#     transforms_list = [tf.Resize(448)]
#     transforms_list.append(tf.RandomHorizontalFlip())
#     transforms_list.append(tf.RandomAffine(degrees=10, translate=(0.1, 0.1)))
#     transforms_list.append(tf.RandomCrop(448))
#     transforms_list.append(tf.ToTensor())
#     transforms_list.append(tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
#     transforms = tf.Compose(transforms_list)

#     dataset = Recipe1M(
#         '/datasets01/recipe1m/012319/',
#         'train',
#         maxnumims=5,
#         maxnumlabels=20,
#         transform=transforms,
#         use_lmdb=True,  ## TODO: true at least for recipe1m
#         shuffle_labels=False,
#         split_data=split_data,
#         include_eos=False,
#         return_img=False,
#         return_ingr=True,
#         return_recipe=True)

#     data_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=100,
#         shuffle=False,
#         num_workers=0,
#         drop_last=False,
#         pin_memory=True,
#         collate_fn=_collate_fn)

#     for info in data_loader:
#         inputs = info
