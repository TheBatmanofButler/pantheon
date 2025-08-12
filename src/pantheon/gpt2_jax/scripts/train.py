import jax
import jax.numpy as jnp

import pantheon.gpt2_jax.core.model as model
import pantheon.gpt2_jax.core.config as config
import pantheon.gpt2_jax.data.load as load
import pantheon.gpt2_jax.data.tokenizer as tokenizer

key = jax.random.PRNGKey(1)
key, model_key = jax.random.split(key)
gpt2 = model.GPT2(model_key)

train_dataloader, test_dataloader = load.build_dataloaders(config)

for batch in train_dataloader:
    print(
        " ".join(
            [
                tokenizer.tokenizer.decode(token)
                for sequence in batch
                for token in sequence[0]
            ]
        )
    )

    predictions = model.predict(
        gpt2,
        batch,
        # batch[0],
    )
    print("decoded", tokenizer.tokenizer.decode(predictions))

    # print(
    #     [
    #         tokenizer.tokenizer.decode(prediction[i])
    #         for prediction in predictions
    #         for i in range(len(prediction))
    #     ]
    # )

    break
