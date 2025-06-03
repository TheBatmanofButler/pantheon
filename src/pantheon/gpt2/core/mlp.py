import einops
import torch
import torch.nn as nn

import pantheon.gpt2.core.config as config


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp):
        super().__init__()

        self.W_input = nn.Parameter(torch.empty(d_model, d_mlp))
        nn.init.normal_(
            self.W_input,
            std=config.initialized_std_range,
        )

        self.W_output = nn.Parameter(torch.empty(d_mlp, d_model))
        nn.init.normal_(
            self.W_output,
            std=config.initialized_std_range,
        )

        self.b_input = nn.Parameter(torch.zeros(d_mlp))
        self.b_output = nn.Parameter(torch.zeros(d_model))

    def forward(self, residual_stream):
        pre_nonlinear = einops.einsum(
            residual_stream,
            self.W_input,
            "batch seq_length d_model, d_model d_mlp -> batch seq_length d_mlp",
        )
        post_nonlinear = torch.nn.functional.gelu(pre_nonlinear)
        mlp_output = (
            einops.einsum(
                post_nonlinear,
                self.W_output,
                "batch seq_length d_mlp, d_mlp d_model -> batch seq_length d_model",
            )
            + self.b_output
        )

        return mlp_output
