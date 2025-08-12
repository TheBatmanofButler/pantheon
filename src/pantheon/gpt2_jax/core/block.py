import equinox as eqx

import pantheon.gpt2_jax.core.attention as attention_lib
import pantheon.gpt2_jax.core.mlp as mlp_lib
import pantheon.gpt2_jax.core.layer_norm as layer_norm_lib


class Block(eqx.Module):
    attention: attention_lib.Attention
    mlp: mlp_lib.MLP
    layer_norm_0: layer_norm_lib.LayerNorm
    layer_norm_1: layer_norm_lib.LayerNorm

    def __init__(self, key, d_embedding, d_mlp, context_window, layer_norm_epsilon):
        self.attention = attention_lib.Attention(key, d_embedding)
        self.mlp = mlp_lib.MLP(key, d_embedding, d_mlp)

        self.layer_norm_0 = layer_norm_lib.LayerNorm(
            context_window,
            d_embedding,
            layer_norm_epsilon,
        )
        self.layer_norm_1 = layer_norm_lib.LayerNorm(
            context_window,
            d_embedding,
            layer_norm_epsilon,
        )

    def __call__(self, x):
        x = self.layer_norm_0(x)
        x = self.attention(x)
        x = self.layer_norm_1(x)
        x = self.mlp(x)

        return x
