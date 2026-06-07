"""Synthetic token sequence builders for DTA tests."""

from __future__ import annotations

import torch


def _token_span(length: int, vocab_size: int) -> torch.Tensor:
    tokens = torch.randint(low=0, high=vocab_size, size=(length,), dtype=torch.long)
    return tokens


def build_cot_token_sequences(
    vocab_size: int,
    system_prompt_length: int,
    thinking_token_length: int,
    response_token_length: int,
    turns: int,
) -> list[torch.Tensor]:
    """Generate multi-turn synthetic CoT-like token sequences, where each turn
    accumulates all previous responses in the context.

    Logic:
      - The first turn consists of: system prompt + thinking tokens + response tokens.
      - Each subsequent turn is: the previous "history" (system prompt + all prior responses)
        + new thinking tokens + new response tokens.
      - The thinking_tokens and response_tokens are sampled independently for each turn.
      - The history grows with every turn as the new response is appended.

    Each output sequence starts with the shared system prompt but includes progressively longer
    "conversation context" as prior responses are accumulated. This structure is designed
    for Dynamic Token Alignment (DTA) engine tests to challenge trie construction with both
    common and incremental prefixes.
    """
    if turns <= 0:
        raise ValueError(f"turns must be positive, got {turns}")

    history = _token_span(system_prompt_length, vocab_size)

    sequences: list[torch.Tensor] = []
    for turn_idx in range(turns):
        thinking_tokens = _token_span(thinking_token_length, vocab_size)
        response_tokens = _token_span(response_token_length, vocab_size)
        sequences.append(
            torch.cat(
                [
                    history,
                    thinking_tokens,
                    response_tokens,
                ]
            )
        )
        history = torch.cat([history, response_tokens])
    return sequences
