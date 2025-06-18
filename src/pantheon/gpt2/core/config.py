from dataclasses import dataclass, asdict


@dataclass
class GPT2Config:
    # Core dimensions
    d_embedding: int = 768
    d_vocab: int = 50257

    # Transformer blocks
    num_blocks: int = 12
    num_heads: int = 8
    d_head: int = d_embedding // num_heads

    # Feed-forward network
    d_mlp: int = d_embedding * 4

    # Sequence length
    context_window: int = 256

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    accumulation_steps: int = 1
    activation_recomputation: bool = True

    # Training duration
    epochs: int = 1
    num_sequences_per_batch: int = 16

    # Weight initialization
    initialized_std_range: float = 1 / ((2 * num_blocks) ** 0.5)

    # Normalization
    layer_norm_epsilon: float = 1e-5

    # Training limits (optional)
    max_batches_per_epoch: int | None = None
    limited_dataset_size: int | None = None

    # Dataset
    dataset_path: str = "roneneldan/TinyStories"
    dataset_name: str | None = None
    test_size: int = limited_dataset_size // 10 if limited_dataset_size else 1000

    # Instrumentation
    wandb_entity: str = "the-ganesh-ravichandran-none"
    wandb_project: str = "gpt2"

    memory_dump_path: str = "profiling_data/snapshot.pickle"
    memory_timeline_path: str = "profiling_data/shapes.html"

    performance_profile_path: str = "profiling_data/traces"
    record_shapes: bool = True

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
