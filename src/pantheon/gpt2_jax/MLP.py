import jax
from flax import nnx
from pantheon.gpt2_jax.config import config


class MLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.W_input = nnx.Linear(
            in_features=config.d_embedding,
            out_features=config.d_mlp,
            rngs=rngs,
        )
        self.W_output = nnx.Linear(
            in_features=config.d_mlp,
            out_features=config.d_embedding,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.W_output(
            self.W_input(x),
        )
