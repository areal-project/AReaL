# SPDX-License-Identifier: Apache-2.0
"""Compare a candidate next-token distribution with the native one, per sequence.

Metrics per row: ``kl_native_to_candidate = sum p log(p / q)``, ``total_variation = 0.5 sum |p - q|``,
``top1_agree`` (argmax equality) and ``selected_logprob_delta = log q[t] - log p[t]`` for the
selected token ``t``. Contract: logits are ``[batch, vocab]`` and are compared in float32.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class DistributionMetrics:
    kl_native_to_candidate: torch.Tensor  # [batch]
    total_variation: torch.Tensor  # [batch]
    top1_agree: torch.Tensor  # [batch], bool
    selected_logprob_delta: torch.Tensor  # [batch]


def distribution_metrics(
    *,
    native_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    selected_tokens: torch.Tensor,
) -> DistributionMetrics:
    """Compute the per-row metrics; the native distribution is the reference."""
    if native_logits.shape != candidate_logits.shape or native_logits.ndim != 2:
        raise ValueError(
            f"logits must both be [batch, vocab], got {tuple(native_logits.shape)} and {tuple(candidate_logits.shape)}"
        )
    if tuple(selected_tokens.shape) != (native_logits.shape[0],):
        raise ValueError(
            f"selected_tokens must be [batch], got {tuple(selected_tokens.shape)}"
        )
    native_logp = torch.log_softmax(native_logits.to(torch.float32), dim=-1)
    candidate_logp = torch.log_softmax(candidate_logits.to(torch.float32), dim=-1)
    native_p = native_logp.exp()
    rows = torch.arange(native_logits.shape[0], device=native_logits.device)
    return DistributionMetrics(
        kl_native_to_candidate=(native_p * (native_logp - candidate_logp)).sum(dim=-1),
        total_variation=0.5 * (native_p - candidate_logp.exp()).abs().sum(dim=-1),
        top1_agree=native_logits.argmax(dim=-1) == candidate_logits.argmax(dim=-1),
        selected_logprob_delta=candidate_logp[rows, selected_tokens]
        - native_logp[rows, selected_tokens],
    )
