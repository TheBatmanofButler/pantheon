import collections
import jax
import jax.numpy as jnp
import datasets
import torch


import pantheon.gpt2_jax.data.tokenizer as tokenizer
from pantheon.gpt2_jax.core.config import GPT2Config


def build_dataloaders(config: GPT2Config):
    # Load training and validation data.
    dataset_dict = datasets.load_dataset(
        config.GPT2Config.dataset_path,
        config.GPT2Config.dataset_name,
    )
    train_dataset = dataset_dict[datasets.Split.TRAIN]
    val_dataset = dataset_dict[datasets.Split.VALIDATION]

    #  Each sample will be padded or truncated to the size of the context window.
    def tokenize(sample: str):
        return tokenizer.tokenizer(
            sample["text"],
            max_length=config.GPT2Config.context_window,
            padding="max_length",
            truncation=True,
        )

    # Tokenize datasets.
    train_dataset = train_dataset.select(range(10)).map(
        tokenize,
        batched=True,
    )
    val_dataset = val_dataset.select(range(10)).map(
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
        batch_size=config.GPT2Config.num_sequences_per_batch,
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
