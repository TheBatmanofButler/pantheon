import equinox as eqx
import jax
import jax.numpy as jnp

import pantheon.gpt2_jax.core.config as config
import pantheon.gpt2_jax.data.tokenizer as tokenizer


class GPT2(eqx.Module):
    embed: eqx.nn.Embedding
    # pos_embed: eqx.nn.Embedding

    def __init__(self, key):
        key, embed_key = jax.random.split(key)

        self.embed = eqx.nn.Embedding(
            config.GPT2Config.d_vocab,
            config.GPT2Config.d_embedding,
            key=embed_key,
        )

        # pos_embed = jnp.zeros(
        #     [
        #         config.GPT2Config.context_window,
        #         config.GPT2Config.d_embedding,
        #     ]
        # )
        # print(1)
        # for i in range(len(pos_embed)):
        #     for j in range(len(pos_embed[i])):
        #         angle = i / 10000 ** (2 * (j // 2) / config.GPT2Config.d_vocab)

        #         if j % 2 == 0:
        #             pos_embed = pos_embed.at[i, j].set(jnp.sin(angle))
        #         else:
        #             pos_embed = pos_embed.at[i, j].set(jnp.cos(angle))
        # print(2)
        # self.pos_embed = eqx.nn.Embedding(weight=pos_embed)

    def unembed(self, x):
        x = jnp.dot(x, self.embed.weight.T)
        x = jnp.argmax(x)

        return x

    def __call__(self, sample):
        x = sample["input_ids"]

        x = jax.vmap(self.embed)(x)  # + self.pos_embed(x)
        x = jax.vmap(self.unembed)(x)
        # print("output", x, x.shape)
        # x = x @ self.embed.weight
        # x = jnp.dot(x, self.embed.weight)
        # print(x)
        # x = x @ self.embed.weight.T
        # x = jnp.dot(x, self.embed.weight.T)
        # print(x)

        return x


def predict(model: GPT2, x):
    return jax.vmap(model)(x)
