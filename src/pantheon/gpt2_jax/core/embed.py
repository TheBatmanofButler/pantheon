import equinox as eqx
import jax
import jax.numpy as jnp


class Embed(eqx.Module):
    W: jax.Array

    def __init__(self, key, d_vocab, d_embedding, initialized_std_range):
        key, W_key = jax.random.split(key, 2)
        self.W = (
            jax.random.normal(
                key=W_key,
                shape=(d_vocab, d_embedding),
            )
            * initialized_std_range
        )

    def __call__(self, x):
        return self.W[x]


class PositionalEmbedding(eqx.Module):
    embedding: eqx.nn.Embedding

    def __init__(self, context_window, d_embedding):
        positions = jnp.arange(context_window)[:, None]
        dims = jnp.arange(d_embedding)[None, :]
        angles = positions / 10000 ** (2 * (dims // 2) / d_embedding)

        self.embedding = eqx.nn.Embedding(
            weight=jnp.where(
                dims % 2 == 0,
                jnp.sin(angles),
                jnp.cos(angles),
            )
        )

    def __call__(self, x):
        return self.embedding(x)


class Unembed(eqx.Module):
    W: jax.Array

    def __init__(self, key, d_vocab, d_embedding, initialized_std_range):
        key, W_key = jax.random.split(key, 2)
        self.W = (
            jax.random.normal(
                key=W_key,
                shape=(d_embedding, d_vocab),
            )
            * initialized_std_range
        )

    def __call__(self, x):
        logits = jnp.matmul(x, self.W)

        return logits


class TiedUnembed(eqx.Module):
    W: jax.Array

    def __init__(
        self,
        embedding_weight: jax.Array,
    ):
        self.W = embedding_weight.T

    def __call__(self, x):
        logits = jnp.matmul(x, self.W)

        return logits
