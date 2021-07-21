import torch
from torch import nn

from inv_cooking.config import ImageEncoderConfig
from inv_cooking.models.modules.utils import freeze_fn


class ClipBasedEncoder(nn.Module):
    PREFIX = "clip_"

    def __init__(self, embed_size: int, config: ImageEncoderConfig):
        super().__init__()
        try:
            import clip
        except:
            raise self._clip_no_intalled_error()
        self.model_name = config.model[len(self.PREFIX) :]
        self.downscale = nn.Upsample(size=(224, 224))
        self.clip_encoder, _ = clip.load(self.model_name, device="cpu", jit=False)
        output_size = self._get_output_size(self.clip_encoder)
        if embed_size != output_size:
            self.adapt_head = nn.Linear(output_size, embed_size)
        else:
            self.adapt_head = nn.Identity()
        if config.freeze:
            freeze_fn(self.clip_encoder)

    @staticmethod
    def _get_output_size(model) -> int:
        with torch.no_grad():
            x = torch.zeros(size=(1, 3, 224, 224))
            y = model.encode_image(x)
            return y.reshape((-1)).size(0)

    @staticmethod
    def _clip_no_intalled_error():
        return ValueError(
            f"You need to have CLIP installed: follow instructions at https://github.com/openai/CLIP"
        )

    def forward(self, image: torch.Tensor):
        """
        :param x: tensor of shape (batch_size, 3, height, width)
        :return shape (batch_size, embedding_size, seq_len)
        """
        with torch.no_grad():
            image = self.downscale(image)
        embedding = self.clip_encoder.encode_image(image)
        embedding = self.adapt_head(embedding)
        return embedding.unsqueeze(-1)
