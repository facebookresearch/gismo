import torch

from inv_cooking.config import (
    ImageEncoderConfig,
    ImageEncoderFreezeType,
    IngredientPredictorTransformerConfig,
    IngredientPredictorType,
    OptimizationConfig,
    RecipeGeneratorConfig,
    TaskType,
)
from inv_cooking.training.module import LitInverseCooking


class TestModule:
    """
    High level tests of the lightning module, to check that the plumbing works
    """

    MAX_NUM_LABELS = 20
    MAX_NUM_INSTRUCTIONS = 10
    MAX_INSTRUCTION_LENGTH = 15
    MAX_RECIPE_LEN = MAX_NUM_INSTRUCTIONS * MAX_INSTRUCTION_LENGTH
    INGR_VOCAB_SIZE = 200
    INGR_EOS_VALUE = INGR_VOCAB_SIZE - 1
    RECIPE_VOCAB_SIZE = 300

    @torch.no_grad()
    def test_im2ingr(self):
        torch.manual_seed(0)

        module = LitInverseCooking(
            task=TaskType.im2ingr,
            image_encoder_config=self.default_image_encoder_config(),
            ingr_pred_config=self.default_ar_ingredient_predictor_config(),
            recipe_gen_config=self.default_recipe_generator_config(),
            optim_config=self.default_optimization_config(),
            max_num_labels=self.MAX_NUM_LABELS,
            max_recipe_len=self.MAX_NUM_INSTRUCTIONS * self.MAX_INSTRUCTION_LENGTH,
            ingr_vocab_size=self.INGR_VOCAB_SIZE,
            instr_vocab_size=self.RECIPE_VOCAB_SIZE,
            ingr_eos_value=self.INGR_EOS_VALUE,
        )

        batch_size = 5
        image = torch.randn(size=(batch_size, 3, 224, 224))
        ingredients = torch.randint(
            low=0, high=self.INGR_VOCAB_SIZE, size=(batch_size, self.MAX_NUM_LABELS + 1)
        )
        recipe = None

        # Try building an optimizer
        assert len(module.create_parameter_groups()) == 2
        optimizers, schedulers = module.configure_optimizers()
        assert len(optimizers) == 1
        assert len(schedulers) == 1

        # Try "train" forward pass
        losses, predictions = module(
            split="train",
            image=image,
            ingredients=ingredients,
            recipe=recipe,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 1
        assert losses["label_loss"].shape == torch.Size([])
        assert len(predictions) == 1
        assert predictions[0].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])

        # Try "train" step
        batch = dict(image=image, ingredients=ingredients, recipe=recipe)
        losses = module.training_step(batch=batch, batch_idx=0)
        assert len(losses) == 1
        assert losses["label_loss"].shape == torch.Size([])

        # Try "val" step
        losses = module.validation_step(batch=batch, batch_idx=0)
        assert losses["label_loss"].shape == torch.Size([])
        assert losses["n_samples"] == 5
        assert losses["ingr_gt"].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])
        assert losses["ingr_pred"].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])

        # Try "test" step
        losses = module.test_step(batch=batch, batch_idx=0)
        assert losses["label_loss"].shape == torch.Size([])
        assert losses["n_samples"] == 5
        assert losses["ingr_gt"].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])
        assert losses["ingr_pred"].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])

    @torch.no_grad()
    def test_ingr2recipe(self):
        torch.manual_seed(0)

        module = LitInverseCooking(
            task=TaskType.ingr2recipe,
            image_encoder_config=self.default_image_encoder_config(),
            ingr_pred_config=self.default_ar_ingredient_predictor_config(),
            recipe_gen_config=self.default_recipe_generator_config(),
            optim_config=self.default_optimization_config(),
            max_num_labels=self.MAX_NUM_LABELS,
            max_recipe_len=self.MAX_NUM_INSTRUCTIONS * self.MAX_INSTRUCTION_LENGTH,
            ingr_vocab_size=self.INGR_VOCAB_SIZE,
            instr_vocab_size=self.RECIPE_VOCAB_SIZE,
            ingr_eos_value=self.INGR_EOS_VALUE,
        )

        batch_size = 5
        image = None
        ingredients = torch.randint(
            low=0, high=self.INGR_VOCAB_SIZE, size=(batch_size, self.MAX_NUM_LABELS + 1)
        )
        recipe = torch.randint(
            low=0, high=self.RECIPE_VOCAB_SIZE, size=(batch_size, self.MAX_RECIPE_LEN)
        )

        # Try building an optimizer
        assert len(module.create_parameter_groups()) == 2
        optimizers, schedulers = module.configure_optimizers()
        assert len(optimizers) == 1
        assert len(schedulers) == 1

        # Try "train" forward pass
        losses, predictions = module(
            split="train",
            image=image,
            ingredients=ingredients,
            recipe=recipe,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 1
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert len(predictions) == 1
        assert predictions[0].shape == torch.Size([5, self.MAX_RECIPE_LEN])

        # Try "train" step
        batch = dict(image=image, ingredients=ingredients, recipe=recipe)
        losses = module.training_step(batch=batch, batch_idx=0)
        assert len(losses) == 1
        assert losses["recipe_loss"].shape == torch.Size([5])

        # Try "val" step
        losses = module.validation_step(batch=batch, batch_idx=0)
        assert len(losses) == 2
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert losses["n_samples"] == 5

        # Try "test" step
        losses = module.test_step(batch=batch, batch_idx=0)
        assert len(losses) == 2
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert losses["n_samples"] == 5

    @torch.no_grad()
    def test_im2recipe(self):
        torch.manual_seed(0)
        module = LitInverseCooking(
            task=TaskType.im2recipe,
            image_encoder_config=self.default_image_encoder_config(),
            ingr_pred_config=self.default_ar_ingredient_predictor_config(),
            recipe_gen_config=self.default_recipe_generator_config(),
            optim_config=self.default_optimization_config(),
            max_num_labels=self.MAX_NUM_LABELS,
            max_recipe_len=self.MAX_NUM_INSTRUCTIONS * self.MAX_INSTRUCTION_LENGTH,
            ingr_vocab_size=self.INGR_VOCAB_SIZE,
            instr_vocab_size=self.RECIPE_VOCAB_SIZE,
            ingr_eos_value=self.INGR_EOS_VALUE,
        )

        batch_size = 5
        image = torch.randn(size=(batch_size, 3, 224, 224))
        ingredients = torch.randint(
            low=0, high=self.INGR_VOCAB_SIZE, size=(batch_size, self.MAX_NUM_LABELS + 1)
        )
        recipe = torch.randint(
            low=0, high=self.RECIPE_VOCAB_SIZE, size=(batch_size, self.MAX_RECIPE_LEN)
        )

        # Try building an optimizer
        assert len(module.create_parameter_groups()) == 4
        optimizers, schedulers = module.configure_optimizers()
        assert len(optimizers) == 1
        assert len(schedulers) == 1

        # Try "train" forward pass
        losses, predictions = module(
            split="train",
            image=image,
            ingredients=ingredients,
            recipe=recipe,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 1
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert predictions[0] is None, "No ingredient prediction in training"
        assert predictions[1].shape == torch.Size([5, self.MAX_RECIPE_LEN])

        # Try "train" step
        batch = dict(image=image, ingredients=ingredients, recipe=recipe)
        losses = module.training_step(batch=batch, batch_idx=0)
        assert losses["recipe_loss"].shape == torch.Size([5])

        # Try "valid" forward step
        losses, predictions = module(
            split="val",
            image=image,
            ingredients=ingredients,
            recipe=recipe,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 1
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert predictions[0] is None, "No ingredient prediction in training"
        assert predictions[1].shape == torch.Size([5, self.MAX_RECIPE_LEN])

        # Try "valid" step
        batch = dict(image=image, ingredients=ingredients, recipe=recipe)
        losses = module.validation_step(batch=batch, batch_idx=0)
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert losses["n_samples"] == 5
        assert losses["ingr_gt"].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])
        assert losses["ingr_pred"] is None

        # Try "test" forward step
        losses, predictions = module(
            split="test",
            image=image,
            ingredients=ingredients,
            recipe=recipe,
            compute_losses=True,
            compute_predictions=True,
        )
        assert len(losses) == 2
        assert losses["label_loss"].shape == torch.Size([])
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert predictions[0].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])
        assert predictions[1].shape == torch.Size([5, self.MAX_RECIPE_LEN])

        # Try "test" step
        batch = dict(image=image, ingredients=ingredients, recipe=recipe)
        losses = module.test_step(batch=batch, batch_idx=0)
        assert losses["label_loss"].shape == torch.Size([])
        assert losses["recipe_loss"].shape == torch.Size([5])
        assert losses["n_samples"] == 5
        assert losses["ingr_gt"].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])
        assert losses["ingr_pred"].shape == torch.Size([5, self.MAX_NUM_LABELS + 1])

    @staticmethod
    def default_optimization_config():
        return OptimizationConfig(
            seed=0,
            lr=0.001,
            scale_lr_pretrained=0.01,
            lr_decay_rate=0.99,
            lr_decay_every=1,
            weight_decay=0.0,
            max_epochs=400,
            patience=10,
            sync_batchnorm=False,
            loss_weights={"label_loss": 1.0, "cardinality_loss": 0.0, "eos_loss": 0.0,},
        )

    @staticmethod
    def default_ar_ingredient_predictor_config():
        return IngredientPredictorTransformerConfig(
            model=IngredientPredictorType.tf,
            layers=0,
            embed_size=2048,
            freeze=False,
            load_pretrained_from="",
            with_set_prediction=False,
            n_att=8,
            dropout=0.3,
        )

    @staticmethod
    def default_image_encoder_config():
        return ImageEncoderConfig(
            model="resnet50",
            pretrained=False,
            dropout=0.1,
            freeze=ImageEncoderFreezeType.none,
        )

    @staticmethod
    def default_recipe_generator_config():
        return RecipeGeneratorConfig(
            dropout=0.5,
            embed_size=2048,
            n_att_heads=8,
            layers=2,
            normalize_before=True,
        )
