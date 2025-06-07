import argparse
import torch
import torch.nn as nn

import pantheon.gpt2.core.model as model
import pantheon.gpt2.core.train as train
import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.device as device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", help="Filepath for trained model state dictionary"
    )
    args = parser.parse_args()

    # gpt2 = nn.DataParallel(model.GPT2()).to(device.device)
    gpt2 = model.GPT2().to(device.device)

    trainer = train.Trainer(
        model=gpt2,
        num_sequences_per_batch=config.config["num_sequences_per_batch"],
        epochs=config.config["epochs"],
        save_fn=lambda: torch.save(gpt2.state_dict(), args.output),
    )
    trainer.train()
