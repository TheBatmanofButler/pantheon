import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


import pantheon.gpt2.core.positional as positional
import pantheon.gpt2.core.embed as embed
import pantheon.gpt2.core.layer_norm as layer_norm
import pantheon.gpt2.core.transformer_block as transformer_block
import pantheon.gpt2.core.config as config


class GPT2(nn.Module):
    def __init__(self, config: config.GPT2Config):
        super().__init__()

        self.embed = embed.Embed(
            d_vocab=config.d_vocab,
            d_embedding=config.d_embedding,
            initialized_std_range=config.initialized_std_range,
        )
        self.positional_embed = positional.PositionalEmbed(
            context_window=config.context_window,
            d_embedding=config.d_embedding,
            initialized_std_range=config.initialized_std_range,
        )

        self.blocks = nn.ModuleList(
            [
                transformer_block.TransformerBlock(
                    block_index=block_index,
                    d_embedding=config.d_embedding,
                    d_head=config.d_head,
                    d_mlp=config.d_mlp,
                    initialized_std_range=config.initialized_std_range,
                    layer_norm_epsilon=config.layer_norm_epsilon,
                    num_heads=config.num_heads,
                )
                for block_index in range(config.num_blocks)
            ]
        )
        self.layer_norm_final = layer_norm.LayerNorm(
            d_embedding=config.d_embedding,
            layer_norm_epsilon=config.layer_norm_epsilon,
        )
        self.unembed = embed.Unembed(
            d_embedding=config.d_embedding,
            d_vocab=config.d_vocab,
            initialized_std_range=config.initialized_std_range,
        )

    def forward(self, tokens) -> torch.Tensor:
        residual = self.embed(tokens) + self.positional_embed(tokens)
        for block in self.blocks:
            residual = checkpoint.checkpoint(self._block, residual, block)
        logits = self.unembed(self.layer_norm_final(residual))

        return logits

    def _block(self, residual, block):
        return block(residual)
