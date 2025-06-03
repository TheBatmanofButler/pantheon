import einops
import torch
import torch.nn as nn

import pantheon.gpt2.core.config as config


class PositionalEmbed(nn.Module):
    def __init__(self, context_window, d_model):
        super().__init__()

        self.W_positional = nn.Parameter(torch.empty((context_window, d_model)))
        nn.init.normal_(self.W_positional, std=config.config["initialized_std_range"])

    def forward(self, tokens):
        batch, sequence_length = tokens.shape

        return einops.repeat(
            self.W_positional[:sequence_length],
            "sequence_length d_model -> batch sequence_length d_model",
            batch=batch,
        )
