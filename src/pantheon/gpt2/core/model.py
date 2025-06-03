import torch
import torch.nn as nn

import pantheon.gpt2.core.config as config
import pantheon.gpt2.core.positional as positional
import pantheon.gpt2.core.embed as embed
import pantheon.gpt2.core.layer_norm as layer_norm
import pantheon.gpt2.core.transformer_block as transformer_block
import pantheon.gpt2.core.device as device


class GPT2(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = embed.Embed(
            d_vocab=config.config["d_vocab"],
            d_model=config.config["d_model"],
        )
        self.positional_embed = positional.PositionalEmbed(
            context_window=config.config["context_window"],
            d_model=config.config["d_model"],
        )

        self.blocks = [
            transformer_block.TransformerBlock(
                num_heads=config.config["num_heads"],
                d_model=config.config["d_model"],
                d_head=config.config["d_head"],
                d_vocab=config.config["d_vocab"],
                d_mlp=config.config["d_mlp"],
                layer_norm_epsilon=config.config["layer_norm_epsilon"],
            ).to(device.device)
            for _ in range(config.config["num_blocks"])
        ]
        self.layer_norm_final = layer_norm.LayerNorm(
            d_model=config.config["d_model"],
            layer_norm_epsilon=config.config["layer_norm_epsilon"],
        )
        self.unembed = embed.Unembed(
            d_model=config.config["d_model"],
            d_vocab=config.config["d_vocab"],
        )

    def forward(self, tokens) -> torch.Tensor:
        residual = self.embed(tokens) + self.positional_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.layer_norm_final(residual))

        return logits
