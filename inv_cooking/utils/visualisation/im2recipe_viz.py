import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from inv_cooking.datasets.recipe1m import Recipe1MDataModule
from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.training.image_to_recipe import ImageToRecipe
from inv_cooking.utils.metrics.gpt2_perplexity import PretrainedLanguageModel
from inv_cooking.utils.metrics.ingredient_iou import IngredientIoU
from inv_cooking.utils.visualisation.recipe_utils import (
    format_recipe,
    ingredients_to_text,
    recipe_to_text,
)


@dataclass
class VisualOutput:
    ingr_vocab: Vocabulary
    instr_vocab: Vocabulary
    gt_image: torch.Tensor
    gt_ingredients: torch.Tensor
    gt_subs_ingredients: torch.Tensor
    gt_recipe: torch.Tensor
    pred_ingredients: torch.Tensor
    pred_recipe: torch.Tensor
    pred_recipe_loss: torch.Tensor
    pred_recipe_from_gt: torch.Tensor
    pred_recipe_from_gt_loss: torch.Tensor
    pred_recipe_from_subs: torch.Tensor
    pred_recipe_from_subs_loss: torch.Tensor

    def __len__(self):
        return self.gt_image.size(0)

    def show(self, i: int, language_model: Optional[PretrainedLanguageModel] = None):
        iou = IngredientIoU(
            ingr_vocab=self.ingr_vocab,
            instr_vocab=self.instr_vocab,
        )

        self.display_image(self.gt_image[i])

        print("\nGT INGREDIENTS:")
        self.display_ingredients(self.gt_ingredients[i], self.ingr_vocab)

        print("\nSUBS INGREDIENTS:")
        self.display_ingredients(self.gt_subs_ingredients[i], self.ingr_vocab)

        print("\nPRED INGREDIENTS:")
        self.display_ingredients(self.pred_ingredients[i], self.ingr_vocab)

        self.display_recipe(
            prefix="GT RECIPE",
            text=self.get_recipe_text(self.gt_recipe[i], self.instr_vocab),
            language_model=language_model,
        )
        iou.visusalise_iou(self.gt_ingredients[i], self.gt_recipe[i])

        self.display_recipe(
            prefix=f"RECIPE {self.perplexity(self.pred_recipe_loss[i])}",
            text=self.get_recipe_text(self.pred_recipe[i], self.instr_vocab),
            language_model=language_model,
        )

        self.display_recipe(
            prefix=f"RECIPE from GT {self.perplexity(self.pred_recipe_from_gt_loss[i])}",
            text=self.get_recipe_text(self.pred_recipe_from_gt[i], self.instr_vocab),
            language_model=language_model,
        )
        iou.visusalise_iou(self.gt_ingredients[i], self.pred_recipe_from_gt[i])

        self.display_recipe(
            prefix=f"RECIPE from SUBS {self.perplexity(self.pred_recipe_from_subs_loss[i])}",
            text=self.get_recipe_text(self.pred_recipe_from_subs[i], self.instr_vocab),
            language_model=language_model,
        )
        iou.visusalise_iou(self.gt_subs_ingredients[i], self.pred_recipe_from_subs[i])

    @classmethod
    def perplexity(cls, loss: torch.Tensor):
        return f"(perplexity: {torch.exp(loss).item():.2f})"

    @classmethod
    def display_image(cls, image_tensor: torch.Tensor):
        import matplotlib.pyplot as plt

        if image_tensor.ndim == 3:
            image = cls.tensor_to_image(image_tensor)
            plt.imshow(image)
            plt.axis("off")
        elif image_tensor.ndim == 4:
            num_images = image_tensor.size(0)
            images = [cls.tensor_to_image(t) for t in image_tensor]

            num_columns = 2
            num_rows = int(math.ceil(num_images / num_columns))
            fig, ax = plt.subplots(
                figsize=(4 * num_columns, 4 * num_rows),
                ncols=num_columns,
                nrows=num_rows,
            )
            for i in range(num_images):
                x, y = divmod(i, num_columns)
                ax[x, y].imshow(images[i])
            plt.show()

    @staticmethod
    def tensor_to_image(tensor: torch.Tensor):
        with torch.no_grad():
            sigma = torch.as_tensor(
                (0.229, 0.224, 0.225), dtype=tensor.dtype, device=tensor.device
            ).view(-1, 1, 1)
            mu = torch.as_tensor(
                (0.485, 0.456, 0.406), dtype=tensor.dtype, device=tensor.device
            ).view(-1, 1, 1)
            tensor = (tensor * sigma) + mu
            tensor = tensor.permute((1, 2, 0))
            array = tensor.cpu().detach().numpy()
            array = np.uint8(array * 255)
            return Image.fromarray(array, mode="RGB")

    @classmethod
    def display_ingredients(cls, prediction: torch.Tensor, vocab: Vocabulary):
        ingredient_list = cls.get_ingredients_text(prediction, vocab)
        print(ingredient_list)

    @staticmethod
    def get_ingredients_text(prediction: torch.Tensor, vocab: Vocabulary):
        return ingredients_to_text(prediction, vocab)

    @classmethod
    def get_recipe_text(cls, prediction: torch.Tensor, vocab: Vocabulary):
        text = recipe_to_text(prediction, vocab)
        text = format_recipe(text)
        return text

    @classmethod
    def display_recipe(
        cls, prefix: str, text: str, language_model: Optional[PretrainedLanguageModel]
    ):
        if language_model is not None:
            lm_ppl = language_model.measure_perplexity(text)
            print(f"\n{prefix} (perplexity GPT: {lm_ppl:.2f}):")
        else:
            print(f"\n{prefix}:")
        for line in text.splitlines():
            print(line)


class Im2RecipeVisualiser:
    """
    Utils to visualise the recipes generated by im2recipe
    """

    def __init__(
        self, model: ImageToRecipe, data_module: Recipe1MDataModule,
    ):
        self.model = model
        self.data_module = data_module
        self.model.eval()

    @torch.no_grad()
    def visualise_impact_of_substitutions(self, batch: Optional[dict] = None):
        if batch is None:
            batch = self.sample_input()
        else:
            batch = copy.copy(batch)

        model_device = next(self.model.parameters()).device
        losses_full, (ingr_pred, recipe_full) = self.model(
            image=batch["image"].to(model_device),
            ingredients=batch["ingredients"].to(model_device),
            recipe=batch["recipe"].to(model_device),
            use_ingr_pred=True,
            compute_losses=True,
            compute_predictions=True,
        )
        losses_gt, (_, recipe_from_gt) = self.model(
            image=batch["image"].to(model_device),
            ingredients=batch["ingredients"].to(model_device),
            recipe=batch["recipe"].to(model_device),
            use_ingr_pred=False,
            compute_losses=True,
            compute_predictions=True,
        )
        losses_gt_subs, (_, recipe_from_gt_subs) = self.model(
            image=batch["image"].to(model_device),
            ingredients=batch["substitution"].to(model_device),
            recipe=batch["recipe"].to(model_device),
            use_ingr_pred=False,
            compute_losses=True,
            compute_predictions=True,
        )

        return VisualOutput(
            ingr_vocab=self.data_module.dataset_test.ingr_vocab,
            instr_vocab=self.data_module.dataset_test.get_instr_vocab(),
            gt_image=batch["image"],
            gt_ingredients=batch["ingredients"],
            gt_subs_ingredients=batch["substitution"],
            gt_recipe=batch["recipe"],
            pred_ingredients=ingr_pred.cpu(),
            pred_recipe=recipe_full.cpu(),
            pred_recipe_loss=losses_full["recipe_loss"].cpu(),
            pred_recipe_from_gt=recipe_from_gt.cpu(),
            pred_recipe_from_gt_loss=losses_gt["recipe_loss"].cpu(),
            pred_recipe_from_subs=recipe_from_gt_subs.cpu(),
            pred_recipe_from_subs_loss=losses_gt_subs["recipe_loss"].cpu(),
        )

    def sample_input(self, batch_size: int = 0, skip: int = 0):
        """
        Sample a batch input from the data loader
        """
        loader = self.data_module.test_dataloader(batch_size=batch_size)
        iterator = iter(loader)
        for _ in range(skip):
            next(iterator)
        return next(iterator)

    def sample_output(
        self,
        batch: Optional[dict] = None,
        with_substitutions: bool = False,
        swap_images: bool = False,
        gray_images: bool = False,
    ):
        """
        Sample an output, using the batch as input to generate the outputs
        or generating a new sample input if not provided
        """
        if batch is None:
            batch = self.sample_input()
        else:
            batch = copy.copy(batch)

        # Investigation of the importance of images:
        # - roll images (each input gets the image of the next input)
        # - replace images by gray images
        if swap_images:
            batch["image"] = batch["image"].roll(shifts=[1], dims=[0])
        elif gray_images:
            images = batch["image"]
            batch["image"] = torch.zeros(
                size=images.shape, dtype=images.dtype, device=images.device
            )
        else:
            batch["image"] = batch["image"]

        self.model.eval()
        ingredients = (
            batch["ingredients"] if not with_substitutions else batch["substitution"]
        )
        with torch.no_grad():
            losses, (ingr_predictions, recipe_predictions) = self.model(
                image=batch["image"],
                ingredients=ingredients,
                recipe=batch["recipe"],
                use_ingr_pred=False,
                compute_losses=True,
                compute_predictions=True,
            )
            return batch, losses, ingr_predictions, recipe_predictions

    def display_sample(
        self,
        batch: Dict[str, Any],
        losses: Dict[str, Any],
        ingr_predictions: torch.Tensor,
        recipe_predictions: torch.Tensor,
        start: int = 0,
        limit: int = -1,
    ):
        """
        Display the outputs of the model in terms of text
        """

        num_recipes = recipe_predictions.shape[0]
        ingr_vocab = self.data_module.dataset_test.ingr_vocab
        instr_vocab = self.data_module.dataset_test.get_instr_vocab()

        if limit < 0:
            limit = num_recipes
        else:
            limit = min(limit, num_recipes)

        for i in range(start, start + limit):

            self.display_image(batch["image"][i])

            print("INGREDIENTS (GT):")
            self.display_ingredients(batch["ingredients"][i], ingr_vocab)

            print("INGREDIENTS (SUBS):")
            self.display_ingredients(batch["substitution"][i], ingr_vocab)

            print("RECIPE (GT):")
            self.display_recipe(batch["recipe"][i], instr_vocab)

            if ingr_predictions is not None:
                print("INGREDIENTS (PRED):")
                self.display_ingredients(ingr_predictions[i], ingr_vocab)

            print("RECIPE (PRED):")
            self.display_recipe(recipe_predictions[i], instr_vocab)

    @classmethod
    def display_image(cls, image_tensor: torch.Tensor):
        import matplotlib.pyplot as plt

        if image_tensor.ndim == 3:
            image = cls.tensor_to_image(image_tensor)
            plt.imshow(image)
            plt.axis("off")
        elif image_tensor.ndim == 4:
            num_images = image_tensor.size(0)
            images = [cls.tensor_to_image(t) for t in image_tensor]

            num_columns = 2
            num_rows = int(math.ceil(num_images / num_columns))
            fig, ax = plt.subplots(
                figsize=(4 * num_columns, 4 * num_rows),
                ncols=num_columns,
                nrows=num_rows,
            )
            for i in range(num_images):
                x, y = divmod(i, num_columns)
                ax[x, y].imshow(images[i])
            plt.show()

    @staticmethod
    def tensor_to_image(tensor: torch.Tensor):
        with torch.no_grad():
            sigma = torch.as_tensor(
                (0.229, 0.224, 0.225), dtype=tensor.dtype, device=tensor.device
            ).view(-1, 1, 1)
            mu = torch.as_tensor(
                (0.485, 0.456, 0.406), dtype=tensor.dtype, device=tensor.device
            ).view(-1, 1, 1)
            tensor = (tensor * sigma) + mu
            tensor = tensor.permute((1, 2, 0))
            array = tensor.cpu().detach().numpy()
            array = np.uint8(array * 255)
            return Image.fromarray(array, mode="RGB")

    @staticmethod
    def display_ingredients(prediction: torch.Tensor, vocab: Vocabulary):
        ingredient_list = []
        for i in prediction.cpu().numpy():
            word = vocab.idx2word.get(i)
            if word != "<pad>":
                if isinstance(word, list):
                    ingredient_list.append(word[0])
                else:
                    ingredient_list.append(word)
        print(ingredient_list)

    @classmethod
    def display_recipe(cls, prediction: torch.Tensor, vocab: Vocabulary):
        text = recipe_to_text(prediction, vocab)
        text = format_recipe(text)
        for line in text.splitlines():
            print(line)
