# Core dimensions
d_embedding = 128
d_vocab = 50257

# Transformer blocks
num_blocks = 12
num_heads = 12
d_head = d_embedding // num_heads

# Feed-forward network
d_mlp = d_embedding * 4

# Sequence length
context_window = 256

# Optimization
learning_rate = 1e-3
weight_decay = 1e-2

# Training duration
epochs = 4
num_sequences_per_batch = 32

# Training limits (optional)
max_batches_per_epoch = None
limited_dataset_size = None

# Weight initialization
initialized_std_range = 1 / ((2 * num_blocks) ** 0.5)

# Normalization
layer_norm_epsilon = 1e-5

# Dataset
dataset = "roneneldan/TinyStories"
test_size = limited_dataset_size // 10 if limited_dataset_size else 1000

config = {
    "context_window": context_window,
    "d_embedding": d_embedding,
    "d_head": d_head,
    "d_mlp": d_mlp,
    "d_vocab": d_vocab,
    "dataset": dataset,
    "epochs": epochs,
    "initialized_std_range": initialized_std_range,
    "layer_norm_epsilon": layer_norm_epsilon,
    "learning_rate": learning_rate,
    "limited_dataset_size": limited_dataset_size,
    "max_batches_per_epoch": max_batches_per_epoch,
    "num_blocks": num_blocks,
    "num_heads": num_heads,
    "num_sequences_per_batch": num_sequences_per_batch,
    "test_size": test_size,
    "weight_decay": weight_decay,
}
