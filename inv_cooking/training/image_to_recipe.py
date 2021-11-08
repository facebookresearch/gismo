from typing import Any, Dict, Optional

import torch
from pytorch_lightning.utilities import rank_zero_only

from inv_cooking.config import (
    ImageEncoderConfig,
    IngredientPredictorConfig,
    IngredientTeacherForcingConfig,
    OptimizationConfig,
    PretrainedConfig,
    RecipeGeneratorConfig,
)
from inv_cooking.config.config import IngredientTeacherForcingFlag
from inv_cooking.models.im2recipe import Im2Recipe
from inv_cooking.training.utils import MonitoredMetric, OptimizationGroup, _BaseModule
from inv_cooking.utils.metrics import (
    DistributedAverage,
    DistributedF1,
    DistributedValLosses,
)
from inv_cooking.utils.metrics.gpt2_perplexity import LanguageModelPerplexity
from inv_cooking.utils.metrics.ingredient_iou import IngredientIoU
from inv_cooking.utils.metrics.recipe_features import RecipeFeaturesMetric


class ImageToRecipe(_BaseModule):
    def __init__(
        self,
        image_encoder_config: ImageEncoderConfig,
        ingr_pred_config: IngredientPredictorConfig,
        recipe_gen_config: RecipeGeneratorConfig,
        optim_config: OptimizationConfig,
        pretrained_im2ingr_config: PretrainedConfig,
        ingr_teachforce_config: IngredientTeacherForcingConfig,
        max_num_ingredients: int,
        max_recipe_len: int,
        ingr_vocab_size: int,
        instr_vocab_size: int,
        ingr_eos_value: int,
    ):
        super().__init__()

        self.model = Im2Recipe(
            image_encoder_config,
            ingr_pred_config,
            recipe_gen_config,
            pretrained_im2ingr_config,
            ingr_vocab_size=ingr_vocab_size,
            instr_vocab_size=instr_vocab_size,
            max_num_ingredients=max_num_ingredients,
            max_recipe_len=max_recipe_len,
            ingr_eos_value=ingr_eos_value,
        )

        self.optimization = optim_config
        self.ingr_teachforce = ingr_teachforce_config

        # check if pretrained models
        self.pretrained_imenc = image_encoder_config.pretrained
        self.pretrained_ingrpred = (
            pretrained_im2ingr_config.load_pretrained_from != "None"
        )

        # metrics to track at validation time
        self.o_f1, self.c_f1, self.i_f1 = DistributedF1.create_all(
            which_f1=["o_f1", "c_f1", "i_f1"],
            pad_value=self.model.ingr_vocab_size - 1,
            remove_eos=self.model.ingr_predictor.requires_eos,
            dist_sync_on_step=True,
        )
        self.val_losses = DistributedValLosses(
            weights=self.optimization.loss_weights,
            monitor_ingr_losses=pretrained_im2ingr_config.freeze,
            dist_sync_on_step=True,
        )
        self.perplexity = DistributedAverage()

        # To compute the recipe perplexity from a language model point of view
        self.input_language_evaluators: Dict[str, LanguageModelPerplexity] = {}
        self.input_language_perplexities = torch.nn.ModuleDict()
        self.outut_language_evaluators: Dict[str, LanguageModelPerplexity] = {}
        self.output_language_perplexities = torch.nn.ModuleDict()

        # To compute metrics on if ingredients appear in the recipe, or length of recipes
        self.ingredient_intersection: Optional[IngredientIoU] = None
        self.input_recipe_feature_metrics = torch.nn.ModuleDict()
        self.output_recipe_feature_metrics = torch.nn.ModuleDict()

    def get_monitored_metric(self) -> MonitoredMetric:
        return MonitoredMetric(name="val_perplexity", mode="min")

    def add_input_feature_metric(self, name: str, metric: RecipeFeaturesMetric):
        self.input_recipe_feature_metrics[name] = metric

    def add_output_feature_metric(self, name: str, metric: RecipeFeaturesMetric):
        self.output_recipe_feature_metrics[name] = metric

    def add_input_language_metric(
        self, name: str, evaluator: LanguageModelPerplexity
    ) -> None:
        self.input_language_evaluators[name] = evaluator
        self.input_language_perplexities[name] = DistributedAverage()

    def add_output_language_metric(
        self, name: str, evaluator: LanguageModelPerplexity
    ) -> None:
        self.outut_language_evaluators[name] = evaluator
        self.output_language_perplexities[name] = DistributedAverage()

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        ingredients: Optional[torch.Tensor] = None,
        recipe: Optional[torch.Tensor] = None,
        use_ingr_pred: bool = False,
        compute_losses: bool = False,
        compute_predictions: bool = False,
    ):
        out = self.model(
            image=image,
            target_recipe=recipe,
            target_ingredients=ingredients,
            use_ingr_pred=use_ingr_pred,
            compute_losses=compute_losses,
            compute_predictions=compute_predictions,
        )
        return out[0], out[1:]

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        out = self(
            image=batch["image"],
            ingredients=batch["ingredients"],
            recipe=batch["recipe"],
            compute_losses=True,
            use_ingr_pred=not self.ingr_teachforce.train,
        )
        return out[0]

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._evaluation_step(
            batch,
            use_ingr_pred=not self.ingr_teachforce.val,
            use_ingr_substitutions=False,
            compute_recipe_predictions=False,
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        use_ingr_prediction = (
            self.ingr_teachforce.test == IngredientTeacherForcingFlag.use_predictions
        )
        use_ingr_substitutions = (
            self.ingr_teachforce.test == IngredientTeacherForcingFlag.use_substitutions
        )
        num_evaluators = len(self.outut_language_evaluators) + len(
            self.input_language_evaluators
        )
        num_feature_metrics = len(self.output_recipe_feature_metrics) + len(self.input_recipe_feature_metrics)
        return self._evaluation_step(
            batch,
            use_ingr_pred=use_ingr_prediction,
            use_ingr_substitutions=use_ingr_substitutions,
            compute_recipe_predictions=(num_evaluators + num_feature_metrics) > 0
            or self.ingredient_intersection,
        )

    def _evaluation_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_ingr_pred: bool,
        use_ingr_substitutions: bool,
        compute_recipe_predictions: bool,
    ):
        ingredients = (
            batch["ingredients"]
            if not use_ingr_substitutions
            else batch["substitution"]
        )
        out = self(
            image=batch["image"],
            ingredients=ingredients,
            recipe=batch["recipe"],
            use_ingr_pred=use_ingr_pred,
            compute_predictions=compute_recipe_predictions,
            compute_losses=True,
        )
        out[0]["n_samples"] = batch["recipe"].shape[0]
        out[0]["ingr_pred"] = out[1][0]
        if compute_recipe_predictions:
            recipe_pred = out[1][1]
            if self.ingredient_intersection:
                self.ingredient_intersection.add(ingredients, recipe_pred)
            for name, metric in self.input_recipe_feature_metrics.items():
                metric.add(batch["recipe"])
            for name, metric in self.output_recipe_feature_metrics.items():
                metric.add(recipe_pred)
            for name, evaluator in self.input_language_evaluators.items():
                out[0][name] = evaluator.compute_batch(batch["recipe"])
            for name, evaluator in self.outut_language_evaluators.items():
                out[0][name] = evaluator.compute_batch(recipe_pred)
        out[0]["ingr_gt"] = batch["ingredients"]
        return out[0]

    def training_step_end(self, losses: Dict[str, torch.Tensor]):
        return self.log_training_losses(losses, self.optimization)

    def validation_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def test_step_end(self, step_output: Dict[str, Any]):
        self._evaluation_step_end(step_output)

    def _evaluation_step_end(self, step_output: Dict[str, Any]):
        """
        Update distributed metrics
        """
        # compute ingredient metrics
        if step_output["ingr_pred"] is not None:
            self.o_f1(step_output["ingr_pred"], step_output["ingr_gt"])
            self.c_f1(step_output["ingr_pred"], step_output["ingr_gt"])
            self.i_f1(step_output["ingr_pred"], step_output["ingr_gt"])

        # update recipe metrics
        for name, perplexity_value in self.input_language_perplexities.items():
            perplexity_value(step_output[name])
        for name, perplexity_value in self.output_language_perplexities.items():
            perplexity_value(step_output[name])
        self.perplexity(torch.exp(step_output["recipe_loss"]))

        # update losses
        self.val_losses(step_output)

    def validation_epoch_end(self, out):
        self._eval_epoch_end(split="val")

    def test_epoch_end(self, out):
        self._eval_epoch_end(split="test")

    def _eval_epoch_end(self, split: str):
        if split == "test":
            self._log_ingredient_metrics(split)

        self.log_test_results(f"{split}_perplexity", self.perplexity.compute())

        # print recipe metrics
        if self.ingredient_intersection:
            self.log_test_results(
                f"{split}_ingr_intersection", self.ingredient_intersection.compute()
            )
        for name, metric in self.input_recipe_feature_metrics.items():
            self.log_test_results(f"{split}_{name}", metric.compute())
        for name, metric in self.output_recipe_feature_metrics.items():
            self.log_test_results(f"{split}_{name}", metric.compute())
        for name, perplexity in self.input_language_perplexities.items():
            self.log_test_results(f"{split}_{name}", perplexity.compute())
        for name, perplexity in self.output_language_perplexities.items():
            self.log_test_results(f"{split}_{name}", perplexity.compute())

        val_losses = self.val_losses.compute()
        for k, v in val_losses.items():
            self.log_test_results(f"{split}_{k}", v)

    def _log_ingredient_metrics(self, split):
        use_ingr_prediction = (
            self.ingr_teachforce.test == IngredientTeacherForcingFlag.use_predictions
        )
        if use_ingr_prediction:
            self.log_test_results(f"{split}_c_f1", self.c_f1.compute())
            self.log_test_results(f"{split}_i_f1", self.i_f1.compute())
            self.log_test_results(f"{split}_o_f1", self.o_f1.compute())
        else:
            self.log_test_results(f"{split}_c_f1", 0.0)
            self.log_test_results(f"{split}_i_f1", 0.0)
            self.log_test_results(f"{split}_o_f1", 0.0)

    def log_test_results(self, key: str, value: Any) -> None:
        self.log(key, value)
        self.log_as_hparam(key, value)

    @rank_zero_only
    def log_as_hparam(self, key: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            value = value.item()
        print(f"[HPARAM] {key}: {value}")

    def configure_optimizers(self):
        return self.create_optimizers(
            optim_groups=self.create_optimization_groups(),
            optim_config=self.optimization,
        )

    def create_optimization_groups(self):
        return [
            OptimizationGroup(
                model=self.model.image_encoder,
                pretrained=self.pretrained_imenc,
                name="image encoder",
            ),
            OptimizationGroup(
                model=self.model.ingr_encoder,
                pretrained=False,
                name="ingredient encoder",
            ),
            OptimizationGroup(
                model=self.model.ingr_predictor,
                pretrained=self.pretrained_ingrpred,
                name="ingredient predictor",
            ),
            OptimizationGroup(
                model=self.model.recipe_gen, pretrained=False, name="recipe generator",
            ),
        ]
