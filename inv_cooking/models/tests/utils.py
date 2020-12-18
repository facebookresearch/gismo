from inv_cooking.config import (
    ImageEncoderConfig,
    ImageEncoderFreezeType,
    RecipeGeneratorConfig,
)


class FakeConfig:
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
