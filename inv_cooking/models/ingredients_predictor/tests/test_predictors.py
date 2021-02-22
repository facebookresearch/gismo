import math

import pytest
import torch

from inv_cooking.config import (
    CardinalityPredictionType,
    IngredientPredictorConfig,
    IngredientPredictorCriterion,
)
from inv_cooking.models.ingredients_predictor import create_ingredient_predictor

from .utils import FakeIngredientPredictorConfig


class TestIngredientPredictor:
    def setup_method(self):
        self.vocab_size = 20
        self.batch_size = 5
        self.max_num_ingredients = 10

    def test_ff_model(self):
        expected_output_shape = torch.Size([self.batch_size, self.max_num_ingredients])
        config = FakeIngredientPredictorConfig.ff_config()
        losses, predictions = self._try_predictor(config)

        assert losses["label_loss"] is not None
        assert "cardinality_loss" not in losses
        assert predictions.shape == expected_output_shape
        assert predictions.min() >= 0
        assert predictions.max() < self.vocab_size

    def test_ff_with_all_losses(self):
        config = FakeIngredientPredictorConfig.ff_config()
        for criterion_type in [
            IngredientPredictorCriterion.bce,
            IngredientPredictorCriterion.iou,
            IngredientPredictorCriterion.td,
        ]:
            config.criterion = criterion_type
            losses, predictions = self._try_predictor(config)
            assert losses["label_loss"] is not None

    def test_ff_with_cardinality(self):
        config = FakeIngredientPredictorConfig.ff_config()
        config.cardinality_pred = CardinalityPredictionType.categorical
        losses, predictions = self._try_predictor(config)
        assert losses["label_loss"] is not None
        assert losses["cardinality_loss"] is not None

    @torch.no_grad()
    def test_ar_models(self):
        torch.manual_seed(0)
        all_configs = [
            FakeIngredientPredictorConfig.lstm_config(),
            FakeIngredientPredictorConfig.lstm_config(with_set_prediction=True),
            FakeIngredientPredictorConfig.tf_config(),
            FakeIngredientPredictorConfig.tf_config(with_set_prediction=True),
        ]
        results = [
            {'label_loss': torch.tensor(2.9594)},
            {'label_loss': torch.tensor(1.1755), 'eos_loss': torch.tensor(0.3032)},
            {'label_loss': torch.tensor(4.3514)},
            {'label_loss': torch.tensor(0.8165), 'eos_loss': torch.tensor(1.4536)}
        ]
        expected_prediction_shape = torch.Size([self.batch_size, self.max_num_ingredients + 1])
        for i, config in enumerate(all_configs):
            losses, predictions = self._try_predictor(config, include_eos=True)
            assert torch.allclose(losses["label_loss"], results[i]["label_loss"], atol=1e-4)
            if config.with_set_prediction:
                assert torch.allclose(losses["eos_loss"], results[i]["eos_loss"], atol=1e-4)
            else:
                assert "eos_loss" not in losses
            assert "cardinality_loss" not in losses
            assert predictions.shape == expected_prediction_shape
            assert predictions.min() >= 0
            assert predictions.max() < self.vocab_size

    @pytest.mark.parametrize("with_set_prediction", [True, False])
    def test_vit_model_with_set(self, with_set_prediction: bool):
        torch.manual_seed(0)
        config = FakeIngredientPredictorConfig.vit_config(with_set_prediction)
        losses, predictions = self._try_predictor(
            config,
            include_eos=True,
            seq_len=self.max_num_ingredients + 1)
        assert predictions.shape == torch.Size([self.batch_size, self.max_num_ingredients + 1])
        assert losses["label_loss"] != torch.tensor(math.inf)
        if config.with_set_prediction:
            assert losses["eos_loss"] != torch.tensor(math.inf)

    def _try_predictor(
        self, config: IngredientPredictorConfig, include_eos: bool = False, seq_len: int = 49
    ):
        image_features = torch.randn(size=(self.batch_size, config.embed_size, seq_len))
        max_num_labels = self.max_num_ingredients + 1 if include_eos else self.max_num_ingredients
        label_target = torch.randint(
            low=0, high=self.vocab_size - 1, size=(self.batch_size, max_num_labels)
        )
        model = create_ingredient_predictor(
            config,
            vocab_size=self.vocab_size,
            max_num_ingredients=self.max_num_ingredients,
            eos_value=0,
        )
        return model(
            image_features,
            label_target,
            compute_losses=True,
            compute_predictions=True,
        )
