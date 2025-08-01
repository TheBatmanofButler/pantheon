import equinox as eqx
import jax
import jax.numpy as jnp


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, in_size, out_size, key):
        wkey, bkey = jax.random.split(key)
        self.weight = jax.random.normal(wkey, (out_size, in_size))
        self.bias = jax.random.normal(bkey, (out_size,))

    def __call__(self, x):
        return self.weight @ x + self.bias


def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean(x)
