import copy
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Tuple, List

import torch

from inv_cooking.datasets.recipe1m import Recipe1MDataModule
from inv_cooking.datasets.vocabulary import Vocabulary
from inv_cooking.training.image_to_recipe import ImageToRecipe
from inv_cooking.utils.metrics.gpt2_perplexity import PretrainedLanguageModel
from inv_cooking.utils.metrics.ingredient_iou import IngredientIoU
from inv_cooking.utils.visualisation.recipe_utils import (
    format_recipe,
    ingredients_to_text,
    recipe_to_text,
    tensor_to_image,
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

    def export(self, i: int, full_display: bool = False):
        """
        Export the output for the GISMo module inputs
        """
        if not full_display:
            self.display_image(self.gt_image[i])
            print("GT_INGREDIENTS:\n", sorted(ingredients_to_text(self.gt_ingredients[i], self.ingr_vocab)))
            print("PRED_INGREDIENTS:\n", sorted(ingredients_to_text(self.pred_ingredients[i], self.ingr_vocab)))
        else:
            self.show(i)

        field = {
            # "gt_ingredients": ingredients_to_text(self.gt_ingredients[i], self.ingr_vocab, full_list=True),
            "ingredients": ingredients_to_text(self.pred_ingredients[i], self.ingr_vocab, full_list=True),
        }
        return field

    def show(self, i: int, language_model: Optional[PretrainedLanguageModel] = None):
        """
        Display the predictions of ingredients, recipes, all alongside the image
        """
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
            image = tensor_to_image(image_tensor)
            plt.imshow(image)
            plt.axis("off")
        elif image_tensor.ndim == 4:
            num_images = image_tensor.size(0)
            images = [tensor_to_image(t) for t in image_tensor]

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
            image = tensor_to_image(image_tensor)
            plt.imshow(image)
            plt.axis("off")
        elif image_tensor.ndim == 4:
            num_images = image_tensor.size(0)
            images = [tensor_to_image(t) for t in image_tensor]

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


class InteractiveSubstitutions:
    """
    End-to-end interactions for substitutions
    """
    def __init__(self, model: ImageToRecipe, data_module: Recipe1MDataModule, use_pred_ingr: bool = True):
        self.model = model
        self.data_module = data_module
        self.use_pred_ingr = use_pred_ingr
        self.model.eval()
        self.vocab_instr = data_module.dataset_test.instr_vocab
        self.vocab_ingr = data_module.dataset_test.ingr_vocab
        self.last_batch = None
        self.last_ingredients_tensor: Optional[torch.Tensor] = None
        self.last_ingredients = []
        self.gismo_preprocess_folder = "/private/home/qduval/baharef/inversecooking2.0/inversecooking2.0/preprocessed_data2/"

    def sample_recipe(self, recipe_id_or_index: Union[str, int, None]):
        if isinstance(recipe_id_or_index, str):
            recipe_id: str = recipe_id_or_index
            batch = self.data_module.dataset_test.build_batch_from_recipe_ids([recipe_id])
        elif isinstance(recipe_id_or_index, int):
            recipe_idx = recipe_id_or_index
            batch = self.data_module.dataset_test.build_batch_from_indices([recipe_idx])
        else:
            return

        self.last_batch = batch
        model_device = next(self.model.parameters()).device
        losses, (ingredients, pred_recipes) = self.model(
            image=batch["image"].to(model_device),
            ingredients=batch["ingredients"].to(model_device),
            recipe=batch["recipe"].to(model_device),
            use_ingr_pred=self.use_pred_ingr,
            compute_losses=True,
            compute_predictions=True,
        )
        if not self.use_pred_ingr:
            ingredients = batch["ingredients"]

        # Display the image
        self.display_image(batch["image"][0])

        # Display the ground truth recipe
        print("GROUND TRUTH RECIPE:")
        self.display_ingredients(ingredients_to_text(batch["ingredients"][0], self.vocab_ingr, full_list=False))
        self.display_recipe(batch["recipe"][0])

        # Display the generated recipe
        print("GENERATED RECIPE:")
        self.display_ingredients(ingredients_to_text(ingredients[0], self.vocab_ingr, full_list=False))
        self.display_recipe(pred_recipes.cpu()[0])

        # Keep in memory what is useful for GISMO to run
        self.last_ingredients_tensor = ingredients.cpu()
        self.last_ingredients = ingredients_to_text(ingredients[0], self.vocab_ingr, full_list=True)

    def compute_substitution(self, old_ingredient: str, lookup: bool = False) -> List[str]:
        return self.compute_substitutions([old_ingredient], lookup=lookup)[0]

    def compute_substitutions(self, old_ingredients: List[str], lookup: bool = False) -> List[List[str]]:
        assert isinstance(old_ingredients, list)
        self._export_to_gismo_format(old_ingredients)
        if lookup:
            output_dir = self._run_lookup_frequency()
        else:
            output_dir = self._run_gismo()
        output_file = os.path.join(output_dir, "val_ranks_out.pkl")
        substitution = pickle.load(open(output_file, "rb"))
        return substitution

    def _export_to_gismo_format(self, old_ingredients: List[str]):
        exports = [
            {
                "id": "",
                "text": 'I used $ XXX $ inside instead of $ ' + old_ingredient + ' $ , and everybody loved them !',
                "subs": (old_ingredient, old_ingredient),
                "ingredients": self.last_ingredients
            }
            for old_ingredient in old_ingredients
        ]
        for destination_file in ["val_comments_subs.pkl", "test_comments_subs.pkl"]:
            destination_path = os.path.join(self.gismo_preprocess_folder, destination_file)
            with open(destination_path, "wb") as f:
                pickle.dump(exports, f)

    def _run_gismo(self) -> str:
        # TODO(config)
        base_dir = "/private/home/qduval/project/inversecooking2.0/gismo"
        run_file_path = f"{base_dir}/run_full_inference.sh"
        # TODO(config)
        output_dir = "/private/home/qduval/baharef/out/lr_5e-05_w_decay_0.0001_hidden_300_emb_d_300_dropout-0.25_nlayers_2_nr_400_neg_sampling_regular_with_titels_False_with_set_True_init_emb_random_lambda_0.0_i_1_data_augmentation_False_context_emb_mode_avg_pool_avg_p_augmentation_0.5_filter_False"
        if os.path.exists(run_file_path):
            os.remove(run_file_path)
        with open(run_file_path, "w") as f:
            f.write(f"cd {base_dir}")
            f.write("\n")
            f.write("conda run -n inv_cooking_gismo ")
            f.write("python train.py name=GIN_MLP setup=context-full max_context=43 lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 nr=400 nlayers=2 lambda_=0.0 i=1 init_emb=random with_titles=False with_set=True filter=False")
            f.write("\n")
            f.write("conda run -n inv_cooking_gismo ")
            f.write(f"python to_val_output.py name=GIN_MLP setup=context-full max_context=43 lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 nr=400 nlayers=2 lambda_=0.0 i=1 init_emb=random with_titles=False with_set=True filter=False")
            f.write("\n")
        os.chmod(run_file_path, 777)
        os.system(f"bash {run_file_path} > /dev/null 2>&1")
        return output_dir

    def _run_lookup_frequency(self) -> str:
        # TODO(config)
        base_dir = "/private/home/qduval/project/inversecooking2.0/gismo"
        run_file_path = f"{base_dir}/run_lookup.sh"
        # TODO(config)
        output_dir = "/private/home/qduval/baharef/out/lr_0.0001_w_decay_0.0005_hidden_200_emb_d_300_dropout-0.5_nlayers_2_nr_400_neg_sampling_regular_with_titels_False_with_set_False_init_emb_random_lambda_0.0_i_0_data_augmentation_False_context_emb_mode_avg_pool_avg_p_augmentation_0.5_filter_False/"
        if os.path.exists(run_file_path):
            os.remove(run_file_path)
        with open(run_file_path, "w") as f:
            f.write(f"cd {base_dir}")
            f.write("\n")
            f.write("conda run -n inv_cooking_gismo ")
            f.write("python train.py name=LTFreq setup=context-free max_context=0")
            # f.write("python train.py name=LT setup=context-free max_context=0")
            f.write("\n")
            f.write("conda run -n inv_cooking_gismo ")
            f.write(f"python to_val_output.py name=GIN_MLP setup=context-full max_context=43 lr=0.00005 w_decay=0.0001 hidden=300 emb_d=300 dropout=0.25 nr=400 nlayers=2 lambda_=0.0 i=1 init_emb=random with_titles=False with_set=True filter=False")
            f.write("\n")
        os.chmod(run_file_path, 777)
        os.system(f"bash {run_file_path} > /dev/null 2>&1")
        return output_dir

    def substitute(self, old_ingredient: str, new_ingredient: str):
        self.substitute_all([(old_ingredient, new_ingredient)])

    def substitute_all(self, pairs: List[Tuple[str, str]]):
        subs_ingredients = self.last_ingredients_tensor.clone()
        for old_ingredient, new_ingredient in pairs:
            ingr_a = self.vocab_ingr.word2idx[old_ingredient]
            ingr_b = self.vocab_ingr.word2idx[new_ingredient]
            subs_ingredients[subs_ingredients == ingr_a] = ingr_b
        self.display_ingredients(ingredients_to_text(subs_ingredients[0], self.vocab_ingr, full_list=False))

        batch = self.last_batch
        batch["substitution"] = subs_ingredients
        model_device = next(self.model.parameters()).device
        losses_from_subs, (_, recipe_from_subs) = self.model(
            image=batch["image"].to(model_device),
            ingredients=batch["substitution"].to(model_device),
            recipe=batch["recipe"].to(model_device),
            use_ingr_pred=False,
            compute_losses=True,
            compute_predictions=True,
        )
        recipe_from_subs = recipe_from_subs.cpu()
        image_tensor = batch["image"][0]
        recipe = recipe_from_subs.cpu()[0]
        self.display_image(image_tensor)
        self.display_recipe(recipe)

    def display_image(self, image_tensor):
        import matplotlib.pyplot as plt
        image = tensor_to_image(image_tensor)
        plt.imshow(image)
        plt.axis("off")

    def display_ingredients(self, ingredients: List[str]):
        for ingr in sorted(ingredients):
            print(f"- {ingr}")

    def display_recipe(self, recipe):
        text = recipe_to_text(recipe, self.vocab_instr)
        text = format_recipe(text)
        for line in text.splitlines():
            print(line)
