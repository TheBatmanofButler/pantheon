import argparse
import torch
import torch.distributed.fsdp as fsdp
import os

import pantheon.gpt2.core.model as model

import pantheon.gpt2.core.train as train

import pantheon.gpt2.core.config as config_lib
from pantheon.gpt2.instrumentation.trainer import TrainingMode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output", help="Filepath for trained model state dictionary"
    )
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    print(f"Process {global_rank}: Starting on local_rank {local_rank}")

    torch.cuda.set_device(local_rank)
    current_device = torch.device(f"cuda:{local_rank}")

    print(f"Rank: {local_rank}")

    config = config_lib.GPT2Config()
    gpt2 = model.GPT2(config=config)

    for layer in gpt2.blocks:
        fsdp.fully_shard(
            layer,
            offload_policy=fsdp.CPUOffloadPolicy(False),
        )
    fsdp.fully_shard(
        gpt2,
        offload_policy=fsdp.CPUOffloadPolicy(False),
    )

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

    torch.distributed.destroy_process_group()
