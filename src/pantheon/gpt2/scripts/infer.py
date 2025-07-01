import argparse
import torch

import pantheon.gpt2.core.config as config_lib
import pantheon.gpt2.core.device as device
import pantheon.gpt2.core.model as model
import pantheon.gpt2.core.sample as sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--filepath", help="Filepath of trained model state dictionary"
    )
    parser.add_argument("-p", "--prompt", help="Prompt for GPT-2")
    args = parser.parse_args()

    config = config_lib.GPT2Config()
    gpt2 = model.GPT2(config=config).to(device.device)
    gpt2.load_state_dict(torch.load(args.filepath, weights_only=True))
    gpt2.eval()

    print("prompt", args.prompt)

    sampler = sample.Sampler(
        gpt2,
        context_window=config.context_window,
    )
    sample_text = sampler.sample(args.prompt)
    print(sample_text)
