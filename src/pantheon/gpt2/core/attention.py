import einops
import torch
import torch.nn as nn
import pantheon.gpt2.core.device as device


class Attention(nn.Module):
    def __init__(self, num_heads, d_embedding, d_head, initialized_std_range):
        super().__init__()

        self.W_Q = nn.Parameter(torch.empty((num_heads, d_embedding, d_head)))
        self.W_K = nn.Parameter(torch.empty((num_heads, d_embedding, d_head)))
        self.W_V = nn.Parameter(torch.empty((num_heads, d_embedding, d_head)))
        self.W_output = nn.Parameter(torch.empty((num_heads, d_head, d_embedding)))

        self.b_Q = nn.Parameter(torch.zeros((num_heads, d_head)))
        self.b_K = nn.Parameter(torch.zeros((num_heads, d_head)))
        self.b_V = nn.Parameter(torch.zeros((num_heads, d_head)))
        self.b_output = nn.Parameter(torch.zeros((d_embedding)))

        nn.init.normal_(self.W_Q, std=initialized_std_range * d_head**0.5)
        nn.init.normal_(self.W_K, std=initialized_std_range * d_head**0.5)
        nn.init.normal_(self.W_V, std=initialized_std_range * d_head**0.5)
        nn.init.normal_(self.W_output, std=initialized_std_range * d_head**0.5)

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
                "num_sequences num_tokens d_embedding, num_heads d_embedding d_head -> num_sequences num_tokens num_heads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                residual_stream,
                self.W_K,
                "num_sequences num_tokens d_embedding, num_heads d_embedding d_head -> num_sequences num_tokens num_heads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                residual_stream,
                self.W_V,
                "num_sequences num_tokens d_embedding, num_heads d_embedding d_head -> num_sequences num_tokens num_heads d_head",
            )
            + self.b_V
        )

        attention_scores = einops.einsum(
            q,
            k,
            "num_sequences num_tokens_q num_heads d_head, num_sequences num_tokens_k num_heads d_head -> num_sequences num_heads num_tokens_q num_tokens_k",
        )

        scaled_attention_scores = attention_scores / (self.d_head**0.5)
        attention_scores_masked = self.apply_causal_mask(scaled_attention_scores)

        noise = torch.randn_like(attention_scores_masked) * 0.01
        attention_scores_masked = attention_scores_masked + noise

        attention_pattern = attention_scores_masked.softmax(-1)

        z = einops.einsum(
            v,
            attention_pattern,
            "num_sequences num_tokens num_heads d_head, num_sequences num_heads num_tokens_q num_tokens_k -> num_sequences num_tokens_q num_heads d_head",
        )

        attention_output = (
            einops.einsum(
                z,
                self.W_output,
                "num_sequences num_tokens num_heads d_head, num_heads d_head d_embedding -> num_sequences num_tokens d_embedding",
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
        attention_scores = attention_scores.masked_fill(mask, self.IGNORE)

        return attention_scores
