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
        self.max_num_labels = 10

    def test_ff_model(self):
        expected_output_shape = torch.Size([self.batch_size, self.max_num_labels])
        config = FakeIngredientPredictorConfig.ff_config()
        losses, predictions = self._try_predictor(config)

        assert losses["label_loss"] is not None
        assert "cardinality_loss" not in losses
        assert predictions.shape == expected_output_shape
        assert predictions.min() >= 0
        assert predictions.max() < self.vocab_size

    def test_ar_models(self):
        expected_output_shape = torch.Size([self.batch_size, self.max_num_labels + 1])
        all_configs = [
            FakeIngredientPredictorConfig.lstm_config(),
            FakeIngredientPredictorConfig.lstm_config(with_set_prediction=True),
            FakeIngredientPredictorConfig.tf_config(),
            FakeIngredientPredictorConfig.tf_config(with_set_prediction=True),
        ]
        for config in all_configs:
            losses, predictions = self._try_predictor(config, include_eos=True)
            assert losses["label_loss"] is not None
            assert "cardinality_loss" not in losses
            assert predictions.shape == expected_output_shape
            assert predictions.min() >= 0
            assert predictions.max() < self.vocab_size

    def test_ff_with_cardinality(self):
        config = FakeIngredientPredictorConfig.ff_config()
        config.cardinality_pred = CardinalityPredictionType.categorical
        losses, predictions = self._try_predictor(config)
        assert losses["label_loss"] is not None
        assert losses["cardinality_loss"] is not None

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

    def _try_predictor(
        self, config: IngredientPredictorConfig, include_eos: bool = False
    ):
        image_features = torch.randn(size=(self.batch_size, config.embed_size, 49))
        max_num_labels = self.max_num_labels + 1 if include_eos else self.max_num_labels
        label_target = torch.randint(
            low=0, high=self.vocab_size - 1, size=(self.batch_size, max_num_labels)
        )
        model = create_ingredient_predictor(
            config,
            vocab_size=self.vocab_size,
            max_num_labels=self.max_num_labels,
            eos_value=0,
        )
        return model(
            image_features, label_target, compute_losses=True, compute_predictions=True,
        )
