from inv_cooking.config import (
    CardinalityPredictionType,
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
    IngredientPredictorCriterion,
)


class FakeIngredientPredictorConfig:

    @staticmethod
    def ff_config() -> IngredientPredictorFFConfig:
        return IngredientPredictorFFConfig(
            model="ff",
            embed_size=2048,
            freeze=False,
            load_pretrained_from="",
            cardinality_pred=CardinalityPredictionType.none,
            layers=2,
            dropout=0.0,
            criterion=IngredientPredictorCriterion.bce,
        )

    @staticmethod
    def lstm_config(with_set_prediction: bool = False) -> IngredientPredictorLSTMConfig:
        return IngredientPredictorLSTMConfig(
            model="lstm",
            embed_size=2048,
            freeze=False,  # setting freeze to True will also freeze the image encoder
            load_pretrained_from="",
            with_set_prediction=with_set_prediction,
            dropout=0.1,
        )

    @staticmethod
    def tf_config(with_set_prediction: bool = False) -> IngredientPredictorTransformerConfig:
        return IngredientPredictorTransformerConfig(
            model="tf",
            layers=0,
            embed_size=2048,
            freeze=False,
            load_pretrained_from="",
            with_set_prediction=with_set_prediction,
            n_att=8,
            dropout=0.3,
        )