from functools import partial
import equinox as eqx
import jax

import pantheon.gpt2_jax.core_eqx.config as config
import pantheon.gpt2_jax.core_eqx.embed as embed_lib
import pantheon.gpt2_jax.core_eqx.block as block_lib


class GPT2(eqx.Module):
    embed: embed_lib.Embed
    pos_embed: embed_lib.PositionalEmbedding
    blocks: list[block_lib.Block]
    unembed: embed_lib.Unembed

    def __init__(self, key):
        key, embed_key = jax.random.split(key, 2)
        block_keys = jax.random.split(key, config.gpt2_config.num_blocks)

        self.embed = embed_lib.Embed(
            key=embed_key,
            d_vocab=config.gpt2_config.d_vocab,
            d_embedding=config.gpt2_config.d_embedding,
            initialized_std_range=config.gpt2_config.initialized_std_range,
        )
        self.pos_embed = embed_lib.PositionalEmbedding(
            config.gpt2_config.context_window,
            config.gpt2_config.d_embedding,
        )

        self.blocks = [
            block_lib.Block(
                block_keys[i],
                config.gpt2_config.d_head,
                config.gpt2_config.d_embedding,
                config.gpt2_config.d_mlp,
                config.gpt2_config.context_window,
                config.gpt2_config.layer_norm_epsilon,
            )
            for i in range(config.gpt2_config.num_blocks)
        ]

        self.unembed = embed_lib.Unembed(
            key=embed_key,
            d_vocab=config.gpt2_config.d_vocab,
            d_embedding=config.gpt2_config.d_embedding,
            initialized_std_range=config.gpt2_config.initialized_std_range,
        )

    def __call__(self, sample):
        x = sample[0]

        x = jax.vmap(self.embed)(x) + jax.vmap(self.pos_embed)(x)
        for block in self.blocks:
            x = block(x)

        x = self.unembed(x)

        return x
