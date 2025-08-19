import equinox as eqx
import jax
import jax.numpy as jnp


class Attention(eqx.Module):
    d_head: int
    context_window: int

    W_Q: jax.Array
    W_K: jax.Array
    W_V: jax.Array
    W_O: jax.Array

    b_Q: jax.Array
    b_K: jax.Array
    b_V: jax.Array
    b_O: jax.Array

    def __init__(self, key, d_head, d_embedding, context_window):
        self.d_head = d_head
        self.context_window = context_window

        key, q_key, k_key, v_key, o_key = jax.random.split(key, 5)

        scale = 0.02

        self.W_Q = (
            jax.random.normal(
                key=q_key,
                shape=(d_embedding, d_embedding),
            )
            * scale
        )
        self.W_K = (
            jax.random.normal(
                key=k_key,
                shape=(d_embedding, d_embedding),
            )
            * scale
        )
        self.W_V = (
            jax.random.normal(
                key=v_key,
                shape=(d_embedding, d_embedding),
            )
            * scale
        )
        self.W_O = (
            jax.random.normal(
                key=o_key,
                shape=(d_embedding, d_embedding),
            )
            * scale
        )

        self.b_Q = jnp.zeros((d_embedding,))
        self.b_K = jnp.zeros((d_embedding,))
        self.b_V = jnp.zeros((d_embedding,))
        self.b_O = jnp.zeros((d_embedding,))

    def __call__(self, x):
        q = jnp.matmul(x, self.W_Q) + self.b_Q
        k = jnp.matmul(x, self.W_K) + self.b_K
        v = jnp.matmul(x, self.W_V) + self.b_V

        k_t = k.transpose()

        attention_scores = jnp.matmul(q, k_t)
        attention_scores = attention_scores / jnp.sqrt(self.d_head)

        mask = jnp.tril(jnp.ones_like(attention_scores))
        attention_scores = jnp.where(mask, attention_scores, -jnp.inf)

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attention_output = jnp.matmul(attention_weights, v)

        output = jnp.matmul(attention_output, self.W_O) + self.b_O

        return output
