import torch
import torch.nn as nn
import torch.distributed.fsdp as fsdp
import os


class SimpleModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=16, output_size=4):  # Absolutely tiny
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x


def train_step(model, optimizer, data, target):
    """Single training step"""
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    print(f"Process {global_rank}: Starting on local_rank {local_rank}")

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    model = SimpleModel()
    fsdp.fully_shard(model, offload_policy=fsdp.CPUOffload(offload_params=True))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create absolutely tiny dummy data
    batch_size = 2  # Reduced from 32
    input_size = 8  # Reduced from 512
    output_size = 4  # Reduced from 256

    # Training loop
    model.train()
    for step in range(5):  # Reduced from 10 steps
        # Generate random data
        data = torch.randn(batch_size, input_size, device=local_rank)
        target = torch.randn(batch_size, output_size, device=local_rank)

        # Training step
        loss = train_step(model, optimizer, data, target)

        if local_rank == 0:  # Only print from main process
            print(f"Step {step}, Loss: {loss:.4f}")

    # Cleanup
    torch.distributed.destroy_process_group()
