import torch
import torch.nn as nn
import numpy as np

import tokenizer
import random


class GPT2(nn.Module):
    def __init__(self):
        super().__init__()

        self.d_vocab = tokenizer.NUM_TOKENS

    def forward(self, tokens) -> torch.Tensor:
        logits = torch.zeros((self.d_vocab,))
        logits[random.randrange(self.d_vocab)] = 1

        return logits
