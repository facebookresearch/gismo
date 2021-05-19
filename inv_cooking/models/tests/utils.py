from inv_cooking.config import (
    ImageEncoderConfig,
    RecipeGeneratorConfig,
    PretrainedConfig,
    IngredientTeacherForcingConfig,
    EncoderAttentionType,
)
from inv_cooking.config.config import TitleEncoderConfig


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
    def no_title_encoder_config() -> TitleEncoderConfig:
        return TitleEncoderConfig(
            with_title=False,
            layers=0,
            layer_dim=512,
        )

    @staticmethod
    def with_title_encoder_config() -> TitleEncoderConfig:
        return TitleEncoderConfig(
            with_title=True,
            layers=2,
            layer_dim=512,
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
