import einops
import torch
import torch.nn as nn
import pantheon.gpt2.core.device as device

import pantheon.gpt2.core.config as config


class Attention(nn.Module):
    def __init__(self, num_heads, d_model, d_head, d_vocab):
        super().__init__()

        self.W_Q = nn.Parameter(torch.empty((num_heads, d_model, d_head)))
        nn.init.normal_(self.W_Q, std=config.initialized_std_range)

        self.W_K = nn.Parameter(torch.empty((num_heads, d_model, d_head)))
        nn.init.normal_(self.W_K, std=config.initialized_std_range)

        self.W_V = nn.Parameter(torch.empty((num_heads, d_model, d_head)))
        nn.init.normal_(self.W_V, std=config.initialized_std_range)

        self.W_output = nn.Parameter(torch.empty((num_heads, d_head, d_model)))
        nn.init.normal_(self.W_output, std=config.initialized_std_range)

        self.b_Q = nn.Parameter(torch.empty((num_heads, d_head)))
        self.b_K = nn.Parameter(torch.empty((num_heads, d_head)))
        self.b_V = nn.Parameter(torch.empty((num_heads, d_head)))
        self.b_output = nn.Parameter(torch.empty((d_model)))

        self.d_head = d_head

        self.register_buffer(
            "IGNORE",
            torch.tensor(float("-inf"), dtype=torch.float32, device=device.device),
        )

    def forward(self, residual_stream):
        q = (
            einops.einsum(
                residual_stream,
                self.W_Q,
                "batch seq_length d_model, num_heads d_model d_head -> batch seq_length num_heads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                residual_stream,
                self.W_K,
                "batch seq_length d_model, num_heads d_model d_head -> batch seq_length num_heads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                residual_stream,
                self.W_V,
                "batch seq_length d_model, num_heads d_model d_head -> batch seq_length num_heads d_head",
            )
            + self.b_V
        )

        attention_scores = einops.einsum(
            q,
            k,
            "batch seq_length_q num_heads d_head, batch seq_length_k num_heads d_head -> batch num_heads seq_length_q seq_length_k",
        )

        scaled_attention_scores = attention_scores / (self.d_head**0.5)
        attention_scores_masked = self.apply_causal_mask(scaled_attention_scores)
        attention_pattern = attention_scores_masked.softmax(-1)

        z = einops.einsum(
            v,
            attention_pattern,
            "batch seq_length num_heads d_head, batch num_heads seq_length_q seq_length_k -> batch seq_length_q num_heads d_head",
        )

        attention_output = (
            einops.einsum(
                z,
                self.W_output,
                "batch seq_length_q num_heads d_head, num_heads d_head d_model -> batch seq_length_q d_model",
            )
            + self.b_output
        )

        return attention_output

    def apply_causal_mask(self, attention_scores):
        all_ones = torch.ones(
            attention_scores.size(-2),
            attention_scores.size(-1),
            device=device.device,
        )
        mask = torch.triu(all_ones, diagonal=1).bool()
        attention_scores.masked_fill_(mask, self.IGNORE)

        return attention_scores
