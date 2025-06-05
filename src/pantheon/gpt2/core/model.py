import torch
import torch.nn as nn

import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.positional as positional
import pantheon.gpt2.core.embed as embed
import pantheon.gpt2.core.layer_norm as layer_norm
import pantheon.gpt2.core.transformer_block as transformer_block


class GPT2(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = embed.Embed(
            d_vocab=config.config["d_vocab"],
            d_embedding=config.config["d_embedding"],
        )
        self.positional_embed = positional.PositionalEmbed(
            context_window=config.config["context_window"],
            d_embedding=config.config["d_embedding"],
        )

        self.blocks = nn.ModuleList(
            [
                transformer_block.TransformerBlock(
                    block_index=block_index,
                    num_heads=config.config["num_heads"],
                    d_embedding=config.config["d_embedding"],
                    d_head=config.config["d_head"],
                    d_mlp=config.config["d_mlp"],
                    layer_norm_epsilon=config.config["layer_norm_epsilon"],
                    initialized_std_range=config.config["initialized_std_range"],
                )
                for block_index in range(config.config["num_blocks"])
            ]
        )
        self.layer_norm_final = layer_norm.LayerNorm(
            d_embedding=config.config["d_embedding"],
            layer_norm_epsilon=config.config["layer_norm_epsilon"],
        )
        self.unembed = embed.Unembed(
            d_embedding=config.config["d_embedding"],
            d_vocab=config.config["d_vocab"],
        )

    def forward(self, tokens) -> torch.Tensor:
        residual = self.embed(tokens) + self.positional_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.layer_norm_final(residual))

        return logits
