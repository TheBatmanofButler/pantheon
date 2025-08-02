import jax
import jax.numpy as jnp

import pantheon.gpt2_jax.core.model as model
import pantheon.gpt2_jax.core.config as config
import pantheon.gpt2_jax.data.load as load
import pantheon.gpt2_jax.data.tokenizer as tokenizer

key = jax.random.PRNGKey(0)
key, model_key = jax.random.split(key)
gpt2 = model.GPT2(model_key)

train_dataloader, test_dataloader = load.build_dataloaders(config)

for batch in train_dataloader:
    print(
        [
            tokenizer.tokenizer.decode(token)
            for sequence in batch["input_ids"]
            for token in sequence
        ]
    )

    predictions = model.predict(
        gpt2,
        batch,
    )

    print(
        [
            tokenizer.tokenizer.decode(prediction[i])
            for prediction in predictions
            for i in range(len(prediction))
        ]
    )

    break
