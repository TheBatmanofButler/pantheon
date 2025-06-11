# GPT-2 Implementation

This directory contains a PyTorch implementation of GPT-2, a transformer-based language model. The implementation is modular and includes various components for training, inference, and model architecture.

Huge thanks to [ARENA](https://www.arena.education/) for their [tutorial series](https://arena-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch), got a ton of help from the exercises there.

## Directory Structure

- `core/`: Core model implementation files
  - `attention.py`: Multi-head attention mechanism
  - `embed.py`: Token embedding and unembedding layers
  - `layer_norm.py`: Layer normalization implementation
  - `loss.py`: Loss functions
  - `mlp.py`: Multi-layer perceptron
  - `model.py`: Main GPT-2 model architecture
  - `positional.py`: Positional embedding layer
  - `sample.py`: Text generation/sampling utilities
  - `train.py`: Training loop and utilities
  - `transformer_block.py`: Transformer block implementation
  - `config.py`: Model configuration
  - `device.py`: Barebones device management

- `data/`: Data processing and dataset utilities
- `instrumentation/`: Performance monitoring and logging tools
- `scripts/`: Utility scripts for various tasks
