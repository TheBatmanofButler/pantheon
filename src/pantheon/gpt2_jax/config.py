from ml_collections import config_dict

config = config_dict.ConfigDict()

# Core dimensions
config.d_embedding = 768
config.d_vocab = 50257

# Sequence length
config.context_window = 512

# Transformer blocks
config.num_blocks = 12
config.num_heads = 8
config.d_head = config.d_embedding // config.num_heads

# Feed-forward network
config.d_mlp = config.d_embedding * 4

# Optimization
config.learning_rate = 3e-4
config.weight_decay = 1e-4
config.accumulation_steps = 1
config.activation_recomputation = True

# Training duration
config.epochs = 1
config.num_sequences_per_batch = 16

# Weight initialization
config.initialized_std_range = 1 / ((2 * config.num_blocks) ** 0.5)

# Normalization
config.layer_norm_epsilon = 1e-5

# Training limits (optional)
config.max_batches_per_epoch = None
config.limited_dataset_size = None

# Dataset
config.dataset_path = "roneneldan/TinyStories"
config.dataset_name = None
config.test_size = (
    config.limited_dataset_size // 10 if config.limited_dataset_size else 1000
)

# Instrumentation
config.wandb_entity = "the-ganesh-ravichandran-none"
config.wandb_project = "gpt2"

config.memory_dump_path = "profiling_data/snapshot"
config.memory_timeline_path = "profiling_data/shapes.html"

config.performance_profile_path = "profiling_data/traces"
config.record_shapes = True
