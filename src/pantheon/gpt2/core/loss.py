from torch import Tensor


def get_log_probs(logits: Tensor, tokens: Tensor) -> Tensor:
    log_probs = logits.log_softmax(dim=-1)
    log_probs_for_tokens = (
        log_probs[:, :-1]
        .gather(
            dim=-1,
            index=tokens[:, 1:].unsqueeze(-1),
        )
        .squeeze(-1)
    )

    return log_probs_for_tokens
