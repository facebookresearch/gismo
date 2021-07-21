import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms

from inv_cooking.config import DatasetConfig

from .dataset import LoadingOptions, Recipe1M
from .preprocess import run_dataset_pre_processing


class Recipe1MDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        loading_options: LoadingOptions,
        seed: int = 1234,
        checkpoint=None,
    ):
        super().__init__()
        self.dataset_config = dataset_config
        self.seed = seed
        self.loading_options = loading_options
        self.checkpoint = (
            checkpoint  ## TODO: check how checkpoint is performed in lightning
        )

    def prepare_data(self):
        if not os.path.isdir(self.dataset_config.pre_processing.save_path):
            print("Pre-processing Recipe1M dataset.")
            run_dataset_pre_processing(
                self.dataset_config.path, self.dataset_config.pre_processing
            )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.dataset_train = self._get_dataset("train")
            self.dataset_val = self._get_dataset("val")
            self.title_vocab_size = self.dataset_train.get_title_vocab_size()
            self.ingr_vocab_size = self.dataset_train.get_ingr_vocab_size()
            self.instr_vocab_size = self.dataset_train.get_instr_vocab_size()
            self.ingr_eos_value = self.dataset_train.ingr_eos_value
            print(f"Training set composed of {len(self.dataset_train)} samples.")
            print(f"Validation set composed of {len(self.dataset_val)} samples.")
        elif stage == "test":
            self.dataset_test = self._get_dataset(stage, self.dataset_config.eval_split)
            print(
                f"Eval split: {self.dataset_config.eval_split} composed of {len(self.dataset_test)} samples."
            )
            self.title_vocab_size = self.dataset_train.get_title_vocab_size()
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
            collate_fn=Recipe1M.collate_fn,
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
            collate_fn=Recipe1M.collate_fn,
            worker_init_fn=self._worker_init_fn,
        )
        return data_loader

    def _get_dataset(self, stage: str, which_split: Optional[str] = None):

        # reads the file with ids to use for the corresponding split
        if which_split == "val":
            splits_filename = os.path.join(
                self.dataset_config.splits_path, which_split + ".txt"
            )  ## PROBLEM IS HERE
            with open(splits_filename, "r") as f:
                selected_indices = np.array([int(line.rstrip("\n")) for line in f])
        else:
            selected_indices = None

        dataset = Recipe1M(
            self.dataset_config.path,
            stage,
            filtering=self.dataset_config.filtering,
            transform=self._get_transforms(stage=stage),
            use_lmdb=False,  # TODO - check if necessary
            selected_indices=selected_indices,
            loading=self.loading_options,
            preprocessed_folder=self.dataset_config.pre_processing.save_path,
        )
        return dataset

    def _get_transforms(self, stage: str):
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
        np.random.seed(self.seed + worker_id)
