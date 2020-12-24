import os
import pickle
from dataclasses import dataclass
from typing import Tuple

import lmdb
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from inv_cooking.config import DatasetFilterConfig


@dataclass
class LoadingOptions:
    with_image: bool = False
    with_ingredient: bool = False
    with_ingredient_eos: bool = False
    with_recipe: bool = False

    def need_load(self) -> bool:
        return self.with_image or self.with_ingredient or self.with_recipe


class Recipe1M(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        filtering: DatasetFilterConfig,
        loading: LoadingOptions,
        transform=None,
        use_lmdb: bool = False,
        selected_indices: np.ndarray = None,
    ):
        self.image_dir = os.path.join(data_dir, "images", split)
        self.pre_processed_dir = os.path.join(data_dir, "preprocessed")
        self.split = split
        self.max_num_images = filtering.max_num_images
        self.max_num_labels = filtering.max_num_labels
        self.max_seq_length = (
            filtering.max_num_instructions * filtering.max_instruction_length
        )
        self.loading = loading
        self.transform = transform
        self.use_lmdb = use_lmdb

        self.ingr_vocab = []
        self.instr_vocab = []
        self.dataset = []

        # load ingredient voc
        if self.loading.with_ingredient:
            self.ingr_vocab = pickle.load(
                open(
                    os.path.join(
                        self.pre_processed_dir, "final_recipe1m_vocab_ingrs.pkl"
                    ),
                    "rb",
                )
            )

            # remove eos from vocabulary list if not needed
            if not self.loading.with_ingredient_eos:
                self.ingr_vocab.remove_eos()
                self.ingr_pad_value = self.get_ingr_vocab_size() - 1
                self.ingr_eos_value = self.ingr_pad_value
            else:
                self.ingr_pad_value = self.get_ingr_vocab_size() - 1
                self.ingr_eos_value = self.ingr_vocab("<end>")

        # load recipe instructions voc
        if self.loading.with_recipe:
            self.instr_vocab = pickle.load(
                open(
                    os.path.join(
                        self.pre_processed_dir, "final_recipe1m_vocab_toks.pkl"
                    ),
                    "rb",
                )
            )

        # load dataset
        if self.loading.need_load():
            self.dataset = pickle.load(
                open(
                    os.path.join(
                        self.pre_processed_dir, "final_recipe1m_" + split + ".pkl"
                    ),
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
                os.path.join(self.pre_processed_dir, "lmdb_" + split),
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        # get ids of data samples used and prune dataset
        ids = []
        for i, entry in enumerate(self.dataset):
            if len(entry["images"]) == 0:
                continue
            ids.append(i)
        ids = np.array(ids)[selected_indices]
        self.dataset = [self.dataset[i] for i in ids]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[Image.Image, "ingredients", "recipe"]:
        image = self._load_image(index) if self.loading.with_image else None
        ret_ingr = self._load_ingredients(index) if self.loading.with_ingredient else None
        recipee = self._load_recipe(index) if self.loading.with_recipe else None
        return image, ret_ingr, recipee

    def _load_ingredients(self, index: int):
        ingr = self.dataset[index]["ingredients"]

        # ingredients to idx
        true_ingr_idxs = []
        for i in range(len(ingr)):
            true_ingr_idxs.append(self.ingr_vocab(ingr[i]))
        true_ingr_idxs = list(set(true_ingr_idxs))

        if self.loading.with_ingredient_eos:
            true_ingr_idxs.append(self.ingr_vocab("<end>"))

        ret_ingr = true_ingr_idxs + [self.ingr_pad_value] * (
            self.max_num_labels + self.loading.with_ingredient_eos - len(true_ingr_idxs)
        )
        return ret_ingr

    def _load_image(self, index: int):
        paths = self.dataset[index]["images"][0 : self.max_num_images]
        if not paths:
            return torch.zeros(size=(3, 224, 224))  # TODO: ???

        # If several images, select one image
        img_idx = np.random.randint(0, len(paths)) if self.split == "train" else 0
        path = paths[img_idx]
        img_path = os.path.join(
            self.image_dir, path[0], path[1], path[2], path[3], path
        )

        # Load the image
        if self.use_lmdb:
            try:
                with self.image_file.begin(write=False) as txn:
                    image = txn.get(path.encode())
                    image = np.frombuffer(image, dtype=np.uint8)
                    image = np.reshape(image, (256, 256, 3))
                image = Image.fromarray(image.astype("uint8"), "RGB")
            except:
                print("Image id not found in lmdb. Loading jpeg file...")
                image = Image.open(img_path).convert("RGB")
        else:
            image = Image.open(img_path).convert("RGB")

        # Transform the image
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _load_recipe(self, index: int):
        title = self.dataset[index]["title"]
        instructions = self.dataset[index]["tokenized"]
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
        ret_rec = ret_rec[0 : self.max_seq_length]
        ret_rec = ret_rec + [self.instr_vocab("<pad>")] * (
            self.max_seq_length - len(ret_rec)
        )
        return ret_rec

    @staticmethod
    def collate_fn(data):
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
