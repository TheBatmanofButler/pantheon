import torch
import torch.nn as nn

import pantheon.gpt2.core.config as config


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()

        self.W = nn.Parameter(torch.empty(d_vocab, d_model))
        nn.init.normal_(
            tensor=self.W,
            mean=0.0,
            std=config.config["initialized_std_range"],
        )

    def forward(self, tokens):
        return self.W[tokens]


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()

        self.W = nn.Parameter(torch.empty(d_model, d_vocab))
        nn.init.normal_(
            tensor=self.W,
            mean=0.0,
            std=config.config["initialized_std_range"],
        )

    def forward(self, tokens):
        return tokens @ self.W
