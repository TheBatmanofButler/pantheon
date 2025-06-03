import torch.nn as nn
import pantheon.gpt2.core.attention as attention
import pantheon.gpt2.core.mlp as mlp
import pantheon.gpt2.core.layer_norm as layer_norm


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model, layer_norm_epsilon, d_head, d_vocab, d_mlp):
        super().__init__()

        self.layer_norm_1 = layer_norm.LayerNorm(d_model, layer_norm_epsilon)
        self.attention = attention.Attention(num_heads, d_model, d_head, d_vocab)

        self.layer_norm_2 = layer_norm.LayerNorm(d_model, layer_norm_epsilon)
        self.mlp = mlp.MLP(d_model, d_mlp)

    def forward(self, residual_stream):
        residual_stream = (
            self.attention(self.layer_norm_1(residual_stream)) + residual_stream
        )
        output = self.mlp(self.layer_norm_2(residual_stream)) + residual_stream

        return output
