import equinox as eqx
import jax
import jax.numpy as jnp


class Attention(eqx.Module):
    d_head: int

    W_Q: jax.Array
    W_K: jax.Array
    W_V: jax.Array

    b_Q: jax.Array
    b_K: jax.Array
    b_V: jax.Array

    def __init__(self, key, d_embedding):
        self.d_head = d_embedding

        key, q_key, k_key, v_key = jax.random.split(key, 4)

        self.W_Q = jax.random.normal(
            key=q_key,
            shape=(d_embedding, d_embedding),
        )
        self.W_K = jax.random.normal(
            key=k_key,
            shape=(d_embedding, d_embedding),
        )
        self.W_V = jax.random.normal(
            key=v_key,
            shape=(d_embedding, d_embedding),
        )

        self.b_Q = jnp.zeros((d_embedding,))
        self.b_K = jnp.zeros((d_embedding,))
        self.b_V = jnp.zeros((d_embedding,))

    def __call__(self, x):
        q = jnp.matmul(x, self.W_Q) + self.b_Q
        k = jnp.matmul(x, self.W_K) + self.b_K
        v = jnp.matmul(x, self.W_V) + self.b_V

        k_t = k.transpose()

        attention_scores = jnp.matmul(q, k_t)
        attention_scores = attention_scores / jnp.sqrt(self.d_head)
        attention_scores = jax.nn.softmax(attention_scores)

        attention_output = jnp.matmul(attention_scores, v)

        return attention_output
