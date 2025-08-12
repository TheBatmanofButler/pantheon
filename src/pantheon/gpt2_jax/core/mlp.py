import equinox as eqx
import jax
import jax.numpy as jnp


class MLP(eqx.Module):
    l0_W: jax.Array
    l0_b: jax.Array

    l1_W: jax.Array
    l1_b: jax.Array

    def __init__(self, key, d_embedding, d_mlp):
        key, l0_W_key, l0_b_key, l1_W_key, l1_b_key = jax.random.split(key, 5)

        self.l0_W = jax.random.normal(
            key=l0_W_key,
            shape=(d_embedding, d_mlp),
        )
        self.l0_b = jax.random.normal(
            key=l0_b_key,
            shape=(d_mlp,),
        )

        self.l1_W = jax.random.normal(
            key=l1_W_key,
            shape=(d_mlp, d_embedding),
        )
        self.l1_b = jax.random.normal(
            key=l1_b_key,
            shape=(d_embedding,),
        )

    def __call__(self, x):
        x = jnp.matmul(x, self.l0_W) + self.l0_b
        x = jax.nn.gelu(x)
        x = jnp.matmul(x, self.l1_W) + self.l1_b

        return x
