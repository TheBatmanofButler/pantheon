import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained(
    "gpt2",
    # download_mode="force_redownload",
)
tokenizer.eos_token = "<|endoftext|>"
