# gpu_mapping_test.py
import torch
import os


def show_gpu_mapping():
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_count = torch.cuda.device_count()

    print(f"Local rank: {local_rank}")
    print(f"Total visible devices: {device_count}")

    torch.cuda.set_device(local_rank)
    props = torch.cuda.get_device_properties(local_rank)

    print(f"Using device {local_rank}:")
    print(f"  Name: {props.name}")
    print(f"  Total memory: {props.total_memory / 1e9:.1f}GB")
    print(
        f"  Free memory: {(props.total_memory - torch.cuda.memory_reserved()) / 1e9:.1f}GB"
    )


if __name__ == "__main__":
    show_gpu_mapping()
