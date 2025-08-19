import equinox as eqx
import jax.numpy as jnp
import datasets
import torch


import pantheon.gpt2_jax.data.tokenizer as tokenizer
from pantheon.gpt2_jax.core.config import GPT2Config


def build_dataloaders(config: GPT2Config):
    # Load training and validation data.
    dataset_dict = datasets.load_dataset(
        config.dataset_path,
        config.dataset_name,
    )
    train_dataset = dataset_dict[datasets.Split.TRAIN]
    val_dataset = dataset_dict[datasets.Split.VALIDATION]

    #  Each sample will be padded or truncated to the size of the context window.
    def tokenize(sample: str):
        return tokenizer.tokenizer(
            sample["text"],
            max_length=config.context_window,
            padding="max_length",
            truncation=True,
        )

    NUM_TRAIN_SAMPLES = 65536
    NUM_VAL_SAMPLES = NUM_TRAIN_SAMPLES // 10

    # Tokenize datasets.
    train_dataset = train_dataset.select(range(NUM_TRAIN_SAMPLES)).map(
        tokenize,
        batched=True,
    )
    val_dataset = val_dataset.select(range(NUM_VAL_SAMPLES)).map(
        tokenize,
        batched=True,
    )

    def collate(batch):
        return jnp.array(
            [jnp.array([sample[key] for key in sample]) for sample in batch]
        )

    # Create dataloader iterables for training and validation.
    train_dataset = train_dataset.select_columns(
        [
            "input_ids",
            "attention_mask",
        ]
    ).with_format("torch")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.num_sequences_per_batch,
        collate_fn=collate,
    )

    val_dataset = val_dataset.select_columns(
        [
            "input_ids",
            "attention_mask",
        ]
    ).with_format("torch")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        collate_fn=collate,
    )

    return train_dataloader, val_dataloader


def load_model(model, config: GPT2Config):
    return eqx.tree_deserialise_leaves(config.saved_model_name, model)
