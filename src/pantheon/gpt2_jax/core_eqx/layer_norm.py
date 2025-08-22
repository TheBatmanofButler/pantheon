import equinox as eqx
import jax
import jax.numpy as jnp


class LayerNorm(eqx.Module):
    layer_norm_epsilon: int

    def __init__(self, context_window, d_embedding, layer_norm_epsilon):
        self.layer_norm_epsilon = layer_norm_epsilon
        pass

    def __call__(self, x: jax.Array):
        mean = jnp.mean(x)
        std = jnp.std(x)
        x = (x - mean) / jnp.sqrt(std + self.layer_norm_epsilon)

        return x
