import argparse
import torch

import pantheon.gpt2.core.model as model
import pantheon.gpt2.core.tokenize as tokenize
import pantheon.gpt2.core.sample as sample
import pantheon.gpt2.core.device as device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filepath", help="Filepath of trained model state dictionary"
    )
    parser.add_argument("-p", "--prompt", help="Prompt for GPT-2")
    args = parser.parse_args()

    gpt2 = model.GPT2(len(tokenize.tokenizer)).to(device.device)
    gpt2.load_state_dict(torch.load(args.filepath, weights_only=True))
    gpt2.eval()

    sampler = sample.Sampler(gpt2)
    sample_text = sampler.sample(args.prompt)
    print(sample_text)
