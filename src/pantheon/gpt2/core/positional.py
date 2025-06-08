import einops
import torch
import torch.nn as nn


class PositionalEmbed(nn.Module):
    def __init__(
        self,
        context_window: int,
        d_embedding: int,
        initialized_std_range: float,
    ):
        super().__init__()

        self.d_embedding = d_embedding

        self.W_positional = nn.Parameter(torch.empty((context_window, d_embedding)))
        nn.init.normal_(
            self.W_positional,
            std=initialized_std_range,
        )

    def forward(self, tokens):
        batch, sequence_length = tokens.shape

        return (
            einops.repeat(
                self.W_positional[:sequence_length],
                "sequence_length d_embedding -> batch sequence_length d_embedding",
                batch=batch,
            )
            * self.d_embedding**0.5
        )
