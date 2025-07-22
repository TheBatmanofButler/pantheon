from flax import nnx
from pantheon.gpt2_jax.config import config


class Attention(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.qkv = nnx.LinearGeneral(
            in_features=config.d_embedding,
            out_features=config.num_heads * config.d_embedding * 3,
            rngs=rngs,
        )

        # config.num_heads,
        # in_features=config.d_embedding,
        # qkv_features=config.d_head,
        # out_features=config.d_embedding,
