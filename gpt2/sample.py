from torch.distributions.categorical import Categorical
from model import GPT2

import argparse
import tokenizer
import torch


class Sampler:
    def __init__(self, model):
        self.model = model

    def sample(self, prompt, max_tokens_generated=100):
        self.model.eval()
        tokens = torch.IntTensor(tokenizer.encode(prompt))

        for _ in range(max_tokens_generated):
            logits = self.model(tokens)
            next_token = torch.tensor([Categorical(logits=logits).sample().item()])
            tokens = torch.cat([tokens, next_token])

            if next_token == tokenizer.EOT:
                break

        return tokenizer.decode(tokens.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", help="Prompt to pass into the model")
    args = parser.parse_args()

    gpt2 = GPT2()
    sampler = Sampler(gpt2)
    sample = sampler.sample(prompt=args.prompt)

    print(sample[len(args.prompt) :])
