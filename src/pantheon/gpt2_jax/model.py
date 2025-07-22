import jax
from flax import nnx
from pantheon.gpt2_jax.config import config
from pantheon.gpt2_jax.transformer_block import TransformerBlock


class GPT2(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.embed = nnx.Embed(
            num_embeddings=config.d_vocab,
            features=config.d_embedding,
            rngs=rngs,
        )
        self.positional_embed = nnx.initializers.normal(
            stddev=config.initialized_std_range
        )(
            rngs.params(),
            (config.context_window, config.d_embedding),
        )

        self.block = TransformerBlock(rngs)

    def __call__(self, inputs: jax.Array):
        x = self.embed(inputs) + self.positional_embed(inputs)
        x = self.block(x)

        return x
