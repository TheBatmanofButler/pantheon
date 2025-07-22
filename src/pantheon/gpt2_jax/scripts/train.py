import jax
import jax.numpy as jnp
from flax import nnx
import optax

from datasets import load_dataset, splits
from transformers import GPT2TokenizerFast

import torch
from torch.utils.data import DataLoader

from pantheon.gpt2_jax.config import config
from pantheon.gpt2_jax.model import GPT2

main_rng = nnx.Rngs(params=42)
model = GPT2(rngs=main_rng)
optimizer = optax.adam(
    learning_rate=optax.warmup_exponential_decay_schedule(
        init_value=0,
        peak_value=config.learning_rate,
        warmup_steps=10,
        transition_steps=1,
        decay_rate=0.99,
    )
)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", pad_token="<pad>")


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )


dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split=splits.Split.TRAIN,
)
tokenized_dataset = dataset.map(tokenize_function)
dataloader = DataLoader(tokenized_dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    input_ids = jnp.array(batch["input_ids"], dtype=jnp.int32)
    print(input_ids)
    x = model(input_ids)
    print(x)
    break
    # attention_mask = jnp.array(batch["attention_mask"].numpy())
    # print("attention_mask", attention_mask)
