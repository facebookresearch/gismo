import os
import pickle
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Tuple, Dict

import lmdb
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from inv_cooking.config import DatasetFilterConfig
from inv_cooking.config.dataset import DatasetAblationConfig
from inv_cooking.datasets.vocabulary import Vocabulary


@dataclass
class LoadingOptions:
    with_image: bool = False
    with_ingredient: bool = False
    with_ingredient_eos: bool = False
    with_recipe: bool = False
    with_title: bool = False
    with_id: bool = False
    with_substitutions: bool = False

    def need_load(self) -> bool:
        return (
            self.with_image
            or self.with_ingredient
            or self.with_recipe
            or self.with_title
            or self.with_id
            or self.with_substitutions
        )


class RecipeEntry(NamedTuple):
    image: Any
    ingredients: Any
    title: Any
    recipe: Any
    recipe_id: Any
    subs_ingredients: Any


class Recipe1M(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        filtering: DatasetFilterConfig,
        ablations: DatasetAblationConfig,
        loading: LoadingOptions,
        preprocessed_folder: str,
        transform=None,
        use_lmdb: bool = False,
        selected_indices: np.ndarray = None,
    ):
        self.image_dir = os.path.join(data_dir, "images", split)
        self.pre_processed_dir = preprocessed_folder
        self.split = split
        self.max_num_images = filtering.max_num_images
        self.max_num_labels = filtering.max_num_labels
        self.max_title_seq_len = filtering.max_title_seq_len
        self.max_seq_length = (
            filtering.max_num_instructions * filtering.max_instruction_length
        )
        self.ablations = ablations
        self.loading = loading
        self.transform = transform
        self.use_lmdb = use_lmdb

        self.ingr_vocab = Vocabulary()
        self.instr_vocab = Vocabulary()
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
        else:
            self.ingr_eos_value = 0
            self.ingr_pad_value = 0

        # load title vocabulary
        if self.loading.with_title:
            self.title_vocab = pickle.load(
                open(
                    os.path.join(
                        self.pre_processed_dir, "final_recipe1m_vocab_title.pkl"
                    ),
                    "rb",
                )
            )
        else:
            self.title_vocab = Vocabulary()

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
        dataset_filename = (
            "final_recipe1msubs_"
            if self.loading.with_substitutions
            else "final_recipe1m_"
        )
        if self.loading.need_load():
            self.dataset = pickle.load(
                open(
                    os.path.join(
                        self.pre_processed_dir, dataset_filename + split + ".pkl"
                    ),
                    "rb",
                )
            )
        else:
            raise ValueError(
                """Dataset loader asked to not return images, nor ingredients, nor titles, nor recipes. 
                                Please set either return_images, return_ingr or return_recipe to True."""
            )

        # Use alternate substitution set (for testing purpose)
        if self.ablations.alternate_substitution_set and self.loading.with_substitutions:
            self.load_alternate_substitutions(self.ablations.alternate_substitution_set)

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
            if len(entry["images"]) == 0 and self.loading.with_image:
                continue
            ids.append(i)

        if selected_indices is not None and not self.loading.with_substitutions:
            selected_indices = [s for s in selected_indices if s < len(ids)]
            ids = np.array(ids)[selected_indices]
        self.dataset = [self.dataset[i] for i in ids]

    def load_alternate_substitutions(self, substitution_set: str):
        alternate_substitutions = pickle.load(open(substitution_set, "rb"))
        alternate_substitutions = {
            (recipe_id, self.ingr_vocab(ingr_a), self.ingr_vocab(ingr_b)): ingr_c
            for (recipe_id, ingr_a, ingr_b), ingr_c in alternate_substitutions.items()
        }
        for recipe_entry in self.dataset:
            recipe_id = recipe_entry["id"]
            ingr_a, ingr_b = recipe_entry["substitution"]
            key = recipe_id, self.ingr_vocab(ingr_a), self.ingr_vocab(ingr_b)
            if key in alternate_substitutions:
                ingr_c = alternate_substitutions[key]
                recipe_entry["substitution"] = ingr_a, ingr_c

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> RecipeEntry:
        """
        Return the relevant elements of the recipe (all optional):
        - image of the finished recipe
        - list of ingredients for the recipe (list of index in ingredient vocabulary)
        - title of the recipe (list of index in title vocabulary)
        - recipe instructions (list of index in instruction vocabulary)
        - id of the recipe (integer)
        - list of substituted ingredients (new ingredients proposed by a different model)
        """
        image = self._load_image(index) if self.loading.with_image else None
        ingredients = (
            self.load_ingredients(index) if self.loading.with_ingredient else None
        )
        recipe = self._load_recipe(index) if self.loading.with_recipe else None
        title = self._load_title(index) if self.loading.with_title else None
        id = self._load_id(index) if self.loading.with_id else None

        # In case we are interested in ingredient substitutions, load the substitution
        # and apply it to the list of ingredient to get a new list of ingredients and
        # return both as outputs
        subs_ingredients = None
        if self.loading.with_substitutions:
            old_ingr, new_ingr = self._load_substitution(index)
            subs_ingredients = self._apply_ingredient_substitution(
                ingredients, old_ingr, new_ingr
            )

        return RecipeEntry(
            image=image,
            ingredients=ingredients,
            title=title,
            recipe=recipe,
            recipe_id=id,
            subs_ingredients=subs_ingredients
        )

    def load_ingredients(self, index: int) -> List[int]:
        raw_ingredients = self.dataset[index]["ingredients"]

        # Map string ingredient to their vocabulary index
        true_ingr_idxs = []
        pad_index = self.ingr_vocab(Vocabulary.PAD_TOKEN)
        for i in range(len(raw_ingredients)):
            ingredient_index = self.ingr_vocab(raw_ingredients[i])
            if ingredient_index != pad_index:
                true_ingr_idxs.append(ingredient_index)
        true_ingr_idxs = list(set(true_ingr_idxs))

        # Add EOS if necessary
        if self.loading.with_ingredient_eos:
            true_ingr_idxs.append(self.ingr_vocab("<end>"))

        # Return the ingredients, completed with pad values
        nb_pad_values = (
            self.max_num_labels + self.loading.with_ingredient_eos - len(true_ingr_idxs)
        )
        return true_ingr_idxs + [self.ingr_pad_value] * nb_pad_values

    def _load_image(self, index: int):
        paths = self.dataset[index]["images"][0 : self.max_num_images]
        if not paths:
            return torch.zeros(size=(3, 224, 224))  # TODO - not the right resolution

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

        # Take into account ablations
        if self.ablations.gray_images:
            return torch.zeros(size=image.shape, device=image.device, dtype=image.dtype)

        return image

    def _load_id(self, index: int) -> int:
        id = self.dataset[index]["id"]
        return id

    def _load_substitution(self, index: int) -> Tuple[int, int]:
        old_ingredient, new_ingredient = self.dataset[index]["substitution"]
        old_ingredient = self.ingr_vocab(old_ingredient)
        new_ingredient = self.ingr_vocab(new_ingredient)

        if self.ablations.alternate_substitution_set:
            recipe_id = self.dataset[index]["id"]


        return old_ingredient, new_ingredient

    @staticmethod
    def _apply_ingredient_substitution(
        ingredients: List[int], old_ingr: int, new_ingr: int
    ) -> List[int]:
        return [(ingr if ingr != old_ingr else new_ingr) for ingr in ingredients]

    def _load_title(self, index: int) -> List[int]:
        tokens = self.dataset[index]["title"]
        out = [self.title_vocab(token) for token in tokens]
        out.append(self.title_vocab("<end>"))
        out = out[0 : self.max_title_seq_len]
        pad_count = self.max_title_seq_len - len(out)
        out = out + [self.title_vocab("<pad>")] * pad_count
        return out

    def _load_recipe(self, index: int) -> List[int]:
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
        """
        Merge images, ingredients, recipe and other elements of the dataset
        into a batch, a dictionary of strings to tensors
        """
        img, ingr, title, recipe, id, substitution = zip(*data)
        ret = {}
        if img[0] is not None:
            ret["image"] = torch.stack(img, 0)
        if ingr[0] is not None:
            ret["ingredients"] = torch.tensor(ingr)
        if title[0] is not None:
            ret["title"] = torch.tensor(title)
        if recipe[0] is not None:
            ret["recipe"] = torch.tensor(recipe)
        if id[0] is not None:
            ret["id"] = id
        if substitution[0] is not None:
            ret["substitution"] = torch.tensor(substitution)
        return ret

    def get_ingr_vocab(self):
        return [
            min(w, key=len) if not isinstance(w, str) else w
            for w in self.ingr_vocab.idx2word.values()
        ]  # includes '<pad>' and eventually '<end>'

    def get_title_vocab_size(self):
        return len(self.title_vocab)

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

    def get_recipe_id_to_index_map(self) -> Dict[str, int]:
        return {
            entry["id"]: index
            for index, entry in enumerate(self.dataset)
        }

    def build_batch_from_indices(self, indices: List[int]):
        data = [self[i] for i in indices]
        return self.collate_fn(data)

    def build_batch_from_recipe_ids(self, recipe_ids: List[str]):
        to_index = self.get_recipe_id_to_index_map()
        indices = [to_index[recipe_id] for recipe_id in recipe_ids]
        return self.build_batch_from_indices(indices)
