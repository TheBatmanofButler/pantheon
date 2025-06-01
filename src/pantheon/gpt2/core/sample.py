from torch.distributions.categorical import Categorical

import torch


import pantheon.gpt2.core.tokenize as tokenize
import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.device as device


class Sampler:
    def __init__(self, model):
        self.model = model

    def _prep_tokens_for_training(self, tokens):
        tokens_within_context = tokens[-config.context_window :]
        tokens_with_batch_dimension = tokens_within_context.unsqueeze(0)

        return tokens_with_batch_dimension

    def _get_final_token_logits(self, logits):
        return logits[0, -1]

    def sample(self, prompt, max_tokens_generated=100):
        self.model.eval()
        input_ids = torch.IntTensor(tokenize.tokenizer.encode(prompt)).to(device.device)

        for _ in range(max_tokens_generated):
            logits = self._get_final_token_logits(
                self.model(self._prep_tokens_for_training(input_ids))
            )

            next_token = torch.tensor(
                [Categorical(logits=logits).sample().item()],
                device=device.device,
            )
            input_ids = torch.cat([input_ids, next_token])

            if next_token == tokenize.tokenizer.eos_token_id:
                break

        return tokenize.tokenizer.decode(input_ids.tolist())
