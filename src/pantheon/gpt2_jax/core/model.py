from functools import partial
import equinox as eqx
import jax

import pantheon.gpt2_jax.core.config as config
import pantheon.gpt2_jax.core.embed as embed_lib
import pantheon.gpt2_jax.core.block as block_lib


class GPT2(eqx.Module):
    embed: eqx.nn.Embedding
    pos_embed: embed_lib.PositionalEmbedding
    blocks: list[block_lib.Block]
    unembed: embed_lib.TiedUnembedding

    def __init__(self, key):
        key, embed_key, block_key = jax.random.split(key, 3)

        self.embed = eqx.nn.Embedding(
            config.GPT2Config.d_vocab,
            config.GPT2Config.d_embedding,
            key=embed_key,
        )
        self.pos_embed = embed_lib.PositionalEmbedding(
            config.GPT2Config.context_window,
            config.GPT2Config.d_embedding,
        )

        self.blocks = [
            block_lib.Block(
                block_key,
                config.GPT2Config.d_embedding,
                config.GPT2Config.d_mlp,
                config.GPT2Config.context_window,
                config.GPT2Config.layer_norm_epsilon,
            )
            for _ in range(config.GPT2Config.num_blocks)
        ]

        self.unembed = embed_lib.TiedUnembedding(self.embed.weight)

    def __call__(self, sample):
        x = sample[0]

        x = jax.vmap(self.embed)(x) + jax.vmap(self.pos_embed)(x)
        for block in self.blocks:
            x = block(x)

        x = self.unembed(x)

        return x


@partial(jax.vmap, in_axes=(None, 0))
def predict(model: GPT2, sample):
    return model(sample)
