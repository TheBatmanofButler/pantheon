import tiktoken

encoding = tiktoken.get_encoding("o200k_base")
EOT = encoding.eot_token
NUM_TOKENS = encoding.max_token_value + 1


def encode(prompt: str) -> tiktoken.Encoding:
    return encoding.encode(prompt)


def decode(encoded_prompt: tiktoken.Encoding) -> str:
    return encoding.decode(encoded_prompt)
