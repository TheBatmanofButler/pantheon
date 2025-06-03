import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model, layer_norm_epsilon):
        super().__init__()

        self.W = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))

        self.layer_norm_epsilon = layer_norm_epsilon

    def forward(self, residual):
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (
            residual.var(
                dim=-1,
                keepdim=True,
                unbiased=False,
            )
            + self.layer_norm_epsilon
        ).sqrt()

        residual = (residual - residual_mean) / residual_std
        residual = residual * self.W + self.b

        return residual
