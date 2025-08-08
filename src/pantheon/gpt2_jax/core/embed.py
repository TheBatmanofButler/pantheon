import equinox as eqx
import jax
import jax.numpy as jnp


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


class TiedUnembedding(eqx.Module):
    W: jax.Array

    def __init__(
        self,
        embedding_weight: jax.Array,
    ):
        self.W = embedding_weight.T

    def __call__(self, x):
        x = jnp.dot(x, self.W)
        x = jnp.argmax(x)

        return x
