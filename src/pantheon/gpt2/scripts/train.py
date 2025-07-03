import argparse
import torch

import pantheon.gpt2.core.model as model

import pantheon.gpt2.core.train as train

import pantheon.gpt2.core.config as config_lib
import pantheon.gpt2.core.device as device
from pantheon.gpt2.instrumentation.trainer import TrainingMode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", help="Filepath for trained model state dictionary"
    )
    args = parser.parse_args()

    config = config_lib.GPT2Config()
    gpt2 = model.GPT2(config=config)
    if torch.cuda.device_count() > 1:
        gpt2 = torch.nn.DataParallel(gpt2)
    gpt2.to(device.device)

    trainer = train.Trainer(
        modes=[
            # TrainingMode.MEMORY,
            # TrainingMode.PERFORMANCE,
            TrainingMode.CHECKPOINTED_SAVES,
            TrainingMode.OBSERVABILITY,
        ],
        config=config,
        save_fn=lambda: torch.save(gpt2.state_dict(), args.output),
        model=gpt2,
    )

    trainer.train()
