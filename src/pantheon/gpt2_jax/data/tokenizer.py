import transformers
import pantheon.gpt2_jax.core.config as config


class Tokenizer:
    def __init__(self):
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            config.GPT2Config.tokenizer_path,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token


tokenizer = Tokenizer().tokenizer
