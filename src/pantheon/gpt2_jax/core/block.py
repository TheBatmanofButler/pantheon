import jax

import pantheon.gpt2_jax.core.attention as attention_lib
import pantheon.gpt2_jax.core.mlp as mlp_lib
import pantheon.gpt2_jax.core.layer_norm as layer_norm_lib


def init(key, d_embedding, d_mlp):
    key, attention_key, mlp_key = jax.random.split(key, 3)

    attention_params = attention_lib.init(attention_key, d_embedding)
    mlp_params = mlp_lib.init(mlp_key, d_embedding, d_mlp)

    return {
        "attention": attention_params,
        "mlp": mlp_params,
    }


def forward(params, x, layer_norm_epsilon):
    x = layer_norm_lib.layer_norm(x, layer_norm_epsilon)
    x = attention_lib.forward(params["attention"], x) + x
    x = layer_norm_lib.layer_norm(x, layer_norm_epsilon)
    x = mlp_lib.forward(params["mlp"], x) + x

    return x
