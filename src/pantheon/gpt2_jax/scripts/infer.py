import argparse
import jax
import jax.numpy as jnp

import pantheon.gpt2_jax.data.load as load
import pantheon.gpt2_jax.core.model as model_lib
import pantheon.gpt2_jax.core.config as config
import pantheon.gpt2_jax.data.tokenizer as tokenizer


key = jax.random.PRNGKey(1)
key, model_key = jax.random.split(key)
gpt2 = model_lib.GPT2(model_key)

loaded_gpt2 = load.load_model(gpt2, config.gpt2_config)


def sample(prompt: str, max_tokens_generated=100):
    tokens = tokenizer.tokenizer(prompt)
    input_ids = jnp.array([tokens["input_ids"]])

    for _ in range(max_tokens_generated):
        logits = gpt2(input_ids)
        logits = logits / config.gpt2_config.temperature

        next_token = jnp.argmax(
            -jax.nn.log_softmax(
                logits[-1],
                axis=-1,
            )
        ).reshape(1, 1)
        input_ids = jnp.concat([input_ids, next_token], axis=-1)

        print(tokenizer.tokenizer.decode(next_token[0, 0]))

    return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", help="Prompt for GPT-2")
    args = parser.parse_args()

    sample(args.prompt)
