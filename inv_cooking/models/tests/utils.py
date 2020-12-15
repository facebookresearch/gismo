from inv_cooking.config import (
    ImageEncoderConfig,
    ImageEncoderFreezeType,
    IngredientPredictorFFConfig,
    RecipeGeneratorConfig,
    IngredientPredictorLSTMConfig, IngredientPredictorTransformerConfig,
)


class FakeConfig:
    @staticmethod
    def ingr_pred_ff_config() -> IngredientPredictorFFConfig:
        return IngredientPredictorFFConfig(
            model="ff_bce",
            embed_size=2048,
            freeze=False,
            load_pretrained_from="",
            with_set_prediction=False,
            with_shuffle_labels=False,
            cardinality_pred="",
            layers=2,
            dropout=0.0,
        )

    @staticmethod
    def ingr_pred_lstm_config() -> IngredientPredictorLSTMConfig:
        return IngredientPredictorLSTMConfig(
            model="lstm",
            embed_size=2048,
            freeze=False,  # setting freeze to True will also freeze the image encoder
            load_pretrained_from="",
            with_set_prediction=False,
            with_shuffle_labels=False,
            dropout=0.1,
        )

    @staticmethod
    def ingr_pred_tf_config() -> IngredientPredictorTransformerConfig:
        return IngredientPredictorTransformerConfig(
            model="tf",
            layers=0,
            embed_size=2048,
            freeze=False,
            load_pretrained_from="",
            with_set_prediction=False,
            with_shuffle_labels=False,
            n_att=8,
            dropout=0.3,
        )

    @staticmethod
    def image_encoder_config() -> ImageEncoderConfig:
        return ImageEncoderConfig(
            model="resnet50",
            pretrained=False,
            dropout=0.1,
            freeze=ImageEncoderFreezeType.none,
        )

    @staticmethod
    def recipe_gen_config() -> RecipeGeneratorConfig:
        return RecipeGeneratorConfig(
            dropout=0.1, embed_size=2048, n_att_heads=1, layers=1, normalize_before=True
        )
