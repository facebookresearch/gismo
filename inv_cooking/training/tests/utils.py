# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from inv_cooking.config import (
    EncoderAttentionType,
    ImageEncoderConfig,
    IngredientPredictorTransformerConfig,
    IngredientPredictorType,
    IngredientTeacherForcingConfig,
    OptimizationConfig,
    PretrainedConfig,
    RecipeGeneratorConfig,
)


class _BaseTest:
    """
    High level tests of the lightning module, to check that the plumbing works
    """

    MAX_NUM_INGREDIENTS = 20
    MAX_NUM_INSTRUCTIONS = 10
    MAX_INSTRUCTION_LENGTH = 15
    MAX_RECIPE_LEN = MAX_NUM_INSTRUCTIONS * MAX_INSTRUCTION_LENGTH
    INGR_VOCAB_SIZE = 200
    INGR_EOS_VALUE = INGR_VOCAB_SIZE - 1
    RECIPE_VOCAB_SIZE = 300

    def setup_method(self):
        print()
        print("-" * 50)
        torch.manual_seed(0)

    @staticmethod
    def assert_all_parameters_used(module):
        total_nb_params = 0
        for group in module.create_optimization_groups():
            params = [p for p in group.model.parameters() if p.requires_grad]
            total_nb_params += sum([p.numel() for p in params])
        expected_nb_params = sum(
            [p.numel() for p in module.model.parameters() if p.requires_grad]
        )
        assert expected_nb_params == total_nb_params

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
            loss_weights={
                "label_loss": 1.0,
                "cardinality_loss": 0.0,
                "eos_loss": 0.0,
            },
        )

    @staticmethod
    def default_ar_ingredient_predictor_config():
        return IngredientPredictorTransformerConfig(
            model=IngredientPredictorType.tf,
            layers=0,
            embed_size=2048,
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
            freeze=False,
        )

    @staticmethod
    def default_recipe_generator_config():
        return RecipeGeneratorConfig(
            dropout=0.5,
            embed_size=2048,
            n_att_heads=8,
            tf_enc_layers=0,
            tf_dec_layers=2,
            encoder_attn=EncoderAttentionType.concat,
            activation="relu",
        )

    @staticmethod
    def default_pretrained_config():
        return PretrainedConfig(
            freeze=False,
            load_pretrained_from="None",
        )

    @staticmethod
    def default_ingr_teachforce_config():
        return IngredientTeacherForcingConfig(
            train=True,
            val=True,
            test=False,
        )
