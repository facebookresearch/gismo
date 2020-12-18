import torch

from inv_cooking.config import CardinalityPredictionType, IngredientPredictorFFConfig, IngredientPredictorConfig
from inv_cooking.models.ingredients_predictor import create_ingredient_predictor
from .utils import FakeIngredientPredictorConfig


class TestIngredientPredictor:
    def setup_method(self):
        self.vocab_size = 20
        self.batch_size = 5
        self.max_num_labels = 10

    def test_classic_predictors(self):
        all_configs = [
            (FakeIngredientPredictorConfig.ff_config(), torch.Size([self.batch_size, self.max_num_labels])),
            (FakeIngredientPredictorConfig.lstm_config(), torch.Size([self.batch_size, self.max_num_labels + 1])),
            (FakeIngredientPredictorConfig.tf_config(), torch.Size([self.batch_size, self.max_num_labels + 1])),
        ]

        for config, expected_output_shape in all_configs:
            losses, predictions = self._try_predictor(config)
            assert losses["label_loss"] is not None
            assert "cardinality_loss" not in losses
            assert predictions.shape == expected_output_shape
            assert predictions.min() >= 0
            assert predictions.max() < self.vocab_size

    def test_ff_with_cardinality(self):
        config = IngredientPredictorFFConfig(
            model="ff_bce",
            embed_size=2048,
            freeze=False,
            load_pretrained_from="",
            cardinality_pred=CardinalityPredictionType.categorical,
            layers=2,
            dropout=0.0,
        )
        losses, predictions = self._try_predictor(config)
        assert losses["label_loss"] is not None
        assert losses["cardinality_loss"] is not None

    def _try_predictor(self, config: IngredientPredictorConfig):
        image_features = torch.randn(size=(self.batch_size, config.embed_size, 49))
        label_target = torch.randint(
            low=0, high=self.vocab_size - 1, size=(self.batch_size, self.max_num_labels)
        )
        model = create_ingredient_predictor(
            config,
            vocab_size=self.vocab_size,
            max_num_labels=self.max_num_labels,
            eos_value=self.vocab_size - 1,
        )
        return model(
            image_features, label_target, compute_losses=True, compute_predictions=True,
        )
