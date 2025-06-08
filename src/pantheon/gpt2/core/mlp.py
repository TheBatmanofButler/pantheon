import einops
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        d_embedding: int,
        d_mlp: int,
        initialized_std_range: float,
    ):
        super().__init__()

        self.W_input = nn.Parameter(torch.empty(d_embedding, d_mlp))
        nn.init.normal_(
            self.W_input,
            std=initialized_std_range,
        )

        self.W_output = nn.Parameter(torch.empty(d_mlp, d_embedding))
        nn.init.normal_(
            self.W_output,
            std=initialized_std_range,
        )

        self.b_input = nn.Parameter(torch.zeros(d_mlp))
        self.b_output = nn.Parameter(torch.zeros(d_embedding))

    def forward(self, residual_stream):
        pre_nonlinear = (
            einops.einsum(
                residual_stream,
                self.W_input,
                "batch seq_length d_embedding, d_embedding d_mlp -> batch seq_length d_mlp",
            )
            + self.b_input
        )
        post_nonlinear = torch.nn.functional.gelu(pre_nonlinear)
        mlp_output = (
            einops.einsum(
                post_nonlinear,
                self.W_output,
                "batch seq_length d_mlp, d_mlp d_embedding -> batch seq_length d_embedding",
            )
            + self.b_output
        )

        return mlp_output
