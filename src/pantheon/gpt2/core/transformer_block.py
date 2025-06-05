import torch.nn as nn
import pantheon.gpt2.core.attention as attention
import pantheon.gpt2.core.mlp as mlp
import pantheon.gpt2.core.layer_norm as layer_norm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        block_index,
        num_heads,
        d_embedding,
        layer_norm_epsilon,
        d_head,
        d_mlp,
        initialized_std_range,
    ):
        super().__init__()

        self.block_index = block_index

        self.layer_norm_1 = layer_norm.LayerNorm(d_embedding, layer_norm_epsilon)
        self.attention = attention.Attention(
            num_heads,
            d_embedding,
            d_head,
            initialized_std_range / (2 * num_heads) ** 0.5,
        )

        self.layer_norm_2 = layer_norm.LayerNorm(d_embedding, layer_norm_epsilon)
        self.mlp = mlp.MLP(d_embedding, d_mlp)

    def forward(self, residual_stream):
        attention_out = self.attention(self.layer_norm_1(residual_stream))
        residual_stream = attention_out + residual_stream

        mlp_out = self.mlp(self.layer_norm_2(residual_stream))
        output = mlp_out + residual_stream

        return output
