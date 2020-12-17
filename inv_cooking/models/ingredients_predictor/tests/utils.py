from inv_cooking.config import (
    IngredientPredictorFFConfig,
    IngredientPredictorLSTMConfig,
    IngredientPredictorTransformerConfig,
)


class FakeIngredientPredictorConfig:
    @staticmethod
    def ff_config() -> IngredientPredictorFFConfig:
        return IngredientPredictorFFConfig(
            model="ff_bce",
            embed_size=2048,
            freeze=False,
            load_pretrained_from="",
            cardinality_pred="",
            layers=2,
            dropout=0.0,
        )

    @staticmethod
    def lstm_config() -> IngredientPredictorLSTMConfig:
        return IngredientPredictorLSTMConfig(
            model="lstm",
            embed_size=2048,
            freeze=False,  # setting freeze to True will also freeze the image encoder
            load_pretrained_from="",
            with_set_prediction=False,
            dropout=0.1,
        )

    @staticmethod
    def tf_config() -> IngredientPredictorTransformerConfig:
        return IngredientPredictorTransformerConfig(
            model="tf",
            layers=0,
            embed_size=2048,
            freeze=False,
            load_pretrained_from="",
            with_set_prediction=False,
            n_att=8,
            dropout=0.3,
        )