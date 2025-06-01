import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
tokenizer.eos_token = "<|endoftext|>"
