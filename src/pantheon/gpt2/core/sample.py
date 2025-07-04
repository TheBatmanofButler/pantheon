from torch.distributions.categorical import Categorical

import torch


import pantheon.gpt2.data.tokenize as tokenize
import pantheon.gpt2.core.device as device
import pantheon.gpt2.core.model as model_lib


class Sampler:
    def __init__(
        self,
        model: model_lib.GPT2,
        context_window: int,
    ):
        self.model = model
        self.context_window = context_window

    def _prep_tokens_for_training(self, tokens):
        tokens_within_context = tokens[-self.context_window :]
        tokens_with_batch_dimension = tokens_within_context.unsqueeze(0)

        return tokens_with_batch_dimension

    def _get_final_token_logits(self, logits):
        return logits[0, -1]

    def sample(self, prompt, max_tokens_generated=100, include_prompt=True):
        self.model.eval()
        prompt_tokens = torch.IntTensor(tokenize.tokenizer.encode(prompt)).to(device.device)
        input_ids = prompt_tokens.clone()

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

        if include_prompt:
            return tokenize.tokenizer.decode(input_ids.tolist())
        else:
            # Return only the generated tokens (excluding the prompt)
            generated_tokens = input_ids[len(prompt_tokens):]
            return tokenize.tokenizer.decode(generated_tokens.tolist())
