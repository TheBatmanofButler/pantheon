import datasets
import torch
import torch.utils.data
import transformer_lens.utils

import pantheon.gpt2.data.tokenize as tokenize
import pantheon.gpt2.core.config as config


def build_dataloaders(
    config: config.GPT2Config,
):
    dataset = datasets.load_dataset(
        path=config.dataset_path,
        name=config.dataset_name,
        split="train",
    )
    if config.limited_dataset_size:
        dataset = dataset.select(range(config.limited_dataset_size))

    tokenized_dataset = transformer_lens.utils.tokenize_and_concatenate(
        dataset,
        tokenize.tokenizer,
        column_name="text",
        max_length=config.context_window,
        add_bos_token=True,
        num_proc=4,
    )

    dataset_dict = tokenized_dataset.train_test_split(test_size=config.test_size)
    train_loader = torch.utils.data.DataLoader(
        dataset_dict["train"],
        batch_size=config.num_sequences_per_batch,
        shuffle=True,
        pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset_dict["train"]),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_dict["test"],
        batch_size=config.num_sequences_per_batch,
        shuffle=False,
        pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset_dict["test"]),
    )

    return train_loader, test_loader
