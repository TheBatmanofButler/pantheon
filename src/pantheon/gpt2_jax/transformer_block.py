import jax
from flax import nnx
from pantheon.gpt2_jax.config import config
from pantheon.gpt2_jax.MLP import MLP


class TransformerBlock(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.layer_norm_1 = nnx.LayerNorm(
            num_features=config.d_embedding,
            epsilon=config.layer_norm_epsilon,
            rngs=rngs,
        )
        self.attention = nnx.MultiHeadAttention(
            num_heads=config.num_heads,
            in_features=config.d_embedding,
            qkv_features=config.d_head,
            rngs=rngs,
        )

        self.layer_norm_2 = nnx.LayerNorm(
            num_features=config.d_embedding,
            epsilon=config.layer_norm_epsilon,
            rngs=rngs,
        )
        self.mlp = MLP(rngs=rngs)

    def __call__(self, inputs: jax.Array):
        x = self.layer_norm_1(inputs)
        x = self.attention(x)
        x = self.layer_norm_2(x)
        x = self.mlp(x)

        return x
