import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from inv_cooking.config import DatasetConfig
from .dataset import Recipe1M


class Recipe1MDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        shuffle_labels=False,
        include_eos=False,
        seed=1234,
        checkpoint=None,
        return_img=False,
        return_ingr=False,
        return_recipe=False,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.shuffle_labels = shuffle_labels
        self.include_eos = include_eos
        self.seed = seed
        self.checkpoint = (
            checkpoint  ## TODO: check how checkpoint is performed in lightning
        )
        self.return_img = return_img
        self.return_ingr = return_ingr
        self.return_recipe = return_recipe

    def prepare_data(self):
        if not os.path.isdir(os.path.join(self.dataset_config.path, "preprocessed")):
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
            batch_size=self.dataset_config.loading.batch_size,
            shuffle=False,
            num_workers=self.dataset_config.loading.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=Recipe1M._collate_fn,
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
            batch_size=self.dataset_config.loading.batch_size,
            shuffle=False,
            num_workers=self.dataset_config.loading.num_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=Recipe1M._collate_fn,
            worker_init_fn=self._worker_init_fn,
        )
        return data_loader

    def _get_dataset(self, stage: str):

        # reads the file with ids to use for the corresponding split
        splits_filename = os.path.join(self.dataset_config.splits_path, stage + ".txt")
        with open(splits_filename, "r") as f:
            selected_indices = np.array([int(line.rstrip("\n")) for line in f])

        dataset = Recipe1M(
            self.dataset_config.path,
            stage,
            filtering=self.dataset_config.filtering,
            transform=self._get_transforms(stage=stage),
            use_lmdb=False,  # TODO - check if necessary
            shuffle_labels=self.shuffle_labels,
            selected_indices=selected_indices,
            include_eos=self.include_eos,
            return_img=self.return_img,
            return_ingr=self.return_ingr,
            return_recipe=self.return_recipe,
        )
        return dataset

    def _get_transforms(self, stage):
        pipeline = [transforms.Resize(self.dataset_config.image_resize)]
        if stage == "train":
            pipeline.append(transforms.RandomHorizontalFlip())
            pipeline.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
            pipeline.append(transforms.RandomCrop(self.dataset_config.image_crop_size))
        else:
            pipeline.append(transforms.CenterCrop(self.dataset_config.image_crop_size))
        pipeline.append(transforms.ToTensor())
        pipeline.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )
        return transforms.Compose(pipeline)

    def _worker_init_fn(self, worker_id: int):
        np.random.seed(self.seed)
