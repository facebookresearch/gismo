import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything

from inv_cooking.config import Config
from inv_cooking.utils.checkpointing import (
    list_available_checkpoints,
    select_best_checkpoint,
)
from inv_cooking.utils.metrics.gpt2_perplexity import (
    LanguageModelPerplexity,
    LanguageModelType,
    PerplexityMetricType,
    PretrainedLanguageModel,
)
from inv_cooking.utils.metrics.ingredient_iou import IngredientIoU
from inv_cooking.utils.metrics.recipe_features import RecipeLengthMetric, RecipeVocabDiversity
from .image_to_recipe import ImageToRecipe
from .trainer import create_model, load_data_set


def run_eval(cfg: Config, gpus: int, nodes: int, distributed_mode: str) -> None:
    """
    Evaluate a model using:
    - either the validation set or the test set
    - ground truth as ingredient or not (for im2recipe)
    """

    seed_everything(cfg.optimization.seed)

    checkpoint_dir = cfg.eval_checkpoint_dir
    all_checkpoints = list_available_checkpoints(checkpoint_dir)
    if len(all_checkpoints) == 0:
        raise ValueError(f"Checkpoint {checkpoint_dir} does not exist.")

    # Creating the data module
    data_module = load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("test")

    # Creating the model
    model = create_model(cfg, data_module)
    monitored_metric = model.get_monitored_metric()

    # Adding custom metrics
    if isinstance(model, ImageToRecipe):
        ingr_vocab = data_module.dataset_test.ingr_vocab
        vocab_instructions = data_module.dataset_test.get_instr_vocab()
        model.ingredient_intersection = IngredientIoU(
            ingr_vocab=ingr_vocab, instr_vocab=vocab_instructions,
        )
        model.add_input_feature_metric("input_recipe_length", RecipeLengthMetric(instr_vocab=vocab_instructions))
        model.add_input_feature_metric("input_recipe_diversity", RecipeVocabDiversity(instr_vocab=vocab_instructions))
        model.add_output_feature_metric("recipe_length", RecipeLengthMetric(instr_vocab=vocab_instructions))
        model.add_output_feature_metric("recipe_diversity", RecipeVocabDiversity(instr_vocab=vocab_instructions))
        language_model = PretrainedLanguageModel(LanguageModelType.medium)
        model.add_input_language_metric(
            name="input_gpt_perplexity",
            evaluator=LanguageModelPerplexity(
                vocab_instructions=vocab_instructions,
                pretrained_language_model=language_model,
                metric_type=PerplexityMetricType.full_recipe,
            ),
        )
        model.add_output_language_metric(
            name="gpt_perplexity",
            evaluator=LanguageModelPerplexity(
                vocab_instructions=vocab_instructions,
                pretrained_language_model=language_model,
                metric_type=PerplexityMetricType.full_recipe,
            ),
        )
        '''
        model.add_output_language_metric(
            name="gpt_perplexity_title",
            evaluator=LanguageModelPerplexity(
                vocab_instructions=vocab_instructions,
                pretrained_language_model=language_model,
                metric_type=PerplexityMetricType.title_only,
            )
        )
        model.add_output_language_metric(
            name="gpt_perplexity_instructions",
            evaluator=LanguageModelPerplexity(
                vocab_instructions=vocab_instructions,
                pretrained_language_model=language_model,
                metric_type=PerplexityMetricType.instructions_only,
            )
        )
        model.add_output_language_metric(
            name="gpt_perplexity_cond_instructions",
            evaluator=LanguageModelPerplexity(
                vocab_instructions=vocab_instructions,
                pretrained_language_model=language_model,
                metric_type=PerplexityMetricType.instructions_conditioned_on_title,
            )
        )
        '''

    # Find best checkpoint path
    best_checkpoint = select_best_checkpoint(all_checkpoints, monitored_metric.mode)
    print(f"Using checkpoint {best_checkpoint}")

    # Load the checkpoint
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model.on_load_checkpoint(checkpoint)  # callback of our models
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # Create the training, initializing from the provided checkpoint
    trainer = pl.Trainer(
        gpus=gpus,
        num_nodes=nodes,
        accelerator=distributed_mode,
        benchmark=True,  # increases speed for fixed image sizes
        precision=32,
        progress_bar_refresh_rate=1 if cfg.slurm.partition == "local" else 0,
    )

    # Run the evaluation on the module
    trainer.test(
        model, datamodule=data_module,
    )
