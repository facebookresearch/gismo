from inv_cooking.config import (
    ImageEncoderConfig,
    RecipeGeneratorConfig,
    PretrainedConfig,
    IngredientTeacherForcingConfig,
    EncoderAttentionType,
)


class FakeConfig:
    @staticmethod
    def image_encoder_config() -> ImageEncoderConfig:
        return ImageEncoderConfig(
            model="resnet50",
            pretrained=False,
            dropout=0.1,
            freeze=False,
        )

    @staticmethod
    def recipe_gen_config() -> RecipeGeneratorConfig:
        return RecipeGeneratorConfig(
            dropout=0.1, embed_size=2048, n_att_heads=1,
            tf_enc_layers=0, tf_dec_layers=1,
            encoder_attn=EncoderAttentionType.concat,
            activation="relu",
        )

    @staticmethod
    def pretrained_config() -> PretrainedConfig:
        return PretrainedConfig(
            freeze=False, load_pretrained_from="None",
        )

    @staticmethod
    def ingr_teachforce_config() -> IngredientTeacherForcingConfig:
        return IngredientTeacherForcingConfig(
            train=True, val=True, test=False,
        )
