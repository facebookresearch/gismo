import torch
import torch.nn as nn

from inv_cooking.config.config import TitleEncoderConfig


class TitleEncoder(nn.Module):
    def __init__(self, config: TitleEncoderConfig, title_vocab_size: int, embed_size: int):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=title_vocab_size, embedding_dim=config.layer_dim)
        layers = []
        for _ in range(config.layers):
            layers.append(nn.TransformerEncoderLayer(d_model=config.layer_dim, nhead=8))
        self.stack = nn.Sequential(*layers)
        self.convert = nn.Conv1d(config.layer_dim, embed_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.stack(x)
        x = x.permute((0, 2, 1))
        x = self.convert(x)
        return x
