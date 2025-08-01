import datasets
import torch
import transformers


import pantheon.gpt2_jax.core.config as config

# Load training and validation data.
dataset_dict = datasets.load_dataset(
    config.GPT2Config.dataset_path,
    config.GPT2Config.dataset_name,
)
train_dataset = dataset_dict[datasets.Split.TRAIN]
val_dataset = dataset_dict[datasets.Split.VALIDATION]


# Load tokenizer.
tokenizer = transformers.GPT2Tokenizer.from_pretrained(
    config.GPT2Config.tokenizer_path,
)
tokenizer.pad_token = tokenizer.unk_token


#  Each sample will be padded or truncated to the size of the context window.
def tokenize(sample: str):
    return tokenizer(
        sample["text"],
        max_length=config.GPT2Config.context_window,
        padding="max_length",
        truncation=True,
    )


# Tokenize datasets.
train_dataset = train_dataset.select(range(6)).map(tokenize, batched=True)
val_dataset = val_dataset.select(range(6)).map(tokenize, batched=True)

# Create dataloader iterables for training and validation.
train_dataset = train_dataset.select_columns(
    [
        "input_ids",
        "attention_mask",
    ]
).with_format("torch")
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=len(train_dataset),
)

val_dataset = val_dataset.select_columns(
    [
        "input_ids",
        "attention_mask",
    ]
).with_format("torch")
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=len(val_dataset),
)

for batch in train_dataloader:
    for ind, sample in enumerate(batch["input_ids"]):
        x = tokenizer.decode(sample)
        print(ind, len(sample), x, batch["attention_mask"][ind])
        print()
