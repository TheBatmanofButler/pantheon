import numpy as np
import torch
import torch.nn as nn


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()

        self.weights = torch.nn.Parameter(torch.empty(d_vocab, d_model))
        nn.init.normal_(
            tensor=self.weights,
            mean=0.0,
            std=(1 / np.sqrt(d_vocab)),
        )

    def forward(self, tokens):
        return self.weights[tokens]


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()

        self.weights = torch.nn.Parameter(torch.empty(d_model, d_vocab))
        nn.init.normal_(
            tensor=self.weights,
            mean=0.0,
            std=(1 / np.sqrt(d_vocab)),
        )

    def forward(self, tokens):
        return tokens @ self.weights
