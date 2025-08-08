from functools import partial
import equinox as eqx
import jax

import pantheon.gpt2_jax.core.config as config
import pantheon.gpt2_jax.core.embed as embed_lib


class GPT2(eqx.Module):
    embed: eqx.nn.Embedding
    pos_embed: embed_lib.PositionalEmbedding
    unembed: embed_lib.TiedUnembedding

    def __init__(self, key):
        key, embed_key = jax.random.split(key)

        self.embed = eqx.nn.Embedding(
            config.GPT2Config.d_vocab,
            config.GPT2Config.d_embedding,
            key=embed_key,
        )
        self.pos_embed = embed_lib.PositionalEmbedding(
            config.GPT2Config.context_window,
            config.GPT2Config.d_embedding,
        )
        self.unembed = embed_lib.TiedUnembedding(self.embed.weight)

    def __call__(self, sample):
        return jax.vmap(self.sample_call)(sample)

    def sample_call(self, sample):
        x = sample["input_ids"]

        x = self.embed(x) + self.pos_embed(x)
        x = self.unembed(x)

        return x


@partial(jax.vmap, in_axes=(None, 0))
def predict(model: GPT2, sample):
    return model(sample)
