from dataclasses import dataclass, asdict


@dataclass
class GPT2Config:
    # Core dimensions
    d_embedding: int = 5
    d_vocab: int = 50257

    # Sequence length
    context_window: int = 5

    # Transformer blocks
    num_blocks: int = 12
    num_heads: int = 12
    d_head: int = d_embedding // num_heads

    # Feed-forward network
    d_mlp: int = d_embedding * 4

    # Optimization
    learning_rate: float = 1e-5

    # Training
    num_sequences_per_batch: int = 16

    # Weight initialization
    initialized_std_range: float = 0.02

    # Normalization
    layer_norm_epsilon: float = 1e-5

    # Inference
    temperature: float = 2.0

    # Dataset
    dataset_path: str = "roneneldan/TinyStories"
    dataset_name: str | None = None

    # Tokenizer
    tokenizer_path: str = "openai-community/gpt2"

    # Instrumentation
    wandb_entity: str = "the-ganesh-ravichandran-none"
    wandb_project: str = "gpt2-jax"

    saved_model_name: str = "gpt2.eqx"

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)


gpt2_config = GPT2Config()
