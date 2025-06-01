import torch
import torch.nn as nn

import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.embed as embed


class GPT2(nn.Module):
    def __init__(self, d_vocab):
        super().__init__()

        self.d_vocab = d_vocab

        self.embed = embed.Embed(
            d_vocab=self.d_vocab,
            d_model=config.d_model,
        )
        self.unembed = embed.Unembed(
            d_model=config.d_model,
            d_vocab=self.d_vocab,
        )

    def forward(self, tokens) -> torch.Tensor:
        residual = self.embed(tokens)
        logits = self.unembed(residual)

        return logits
