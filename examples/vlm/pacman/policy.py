# SPDX-License-Identifier: Apache-2.0

"""Single-token constrained policies shared by both Pacman curricula."""

import math
from functools import cached_property
from typing import Any

import torch
import torch.distributed as dist

from areal.api.io_struct import ModelRequest
from areal.api.rl_plugins import GenerationRequestPlugin, PolicyDistribution


class _SumCandidateLogits(torch.autograd.Function):
    """Replicate candidate logits while keeping vocabulary-shard gradients local.

    The loss is replicated across TP ranks, as for vocabulary-parallel cross
    entropy. Backward must NOT sum these identical loss gradients again.
    """

    @staticmethod
    def forward(
        ctx: Any, values: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        result = values.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None


class TokenSetDistribution(PolicyDistribution):
    """Normalize over prediction-aligned ``policy_support`` (token ID + 1).

    Zero denotes padding. Empty support rows are non-policy positions and
    produce zero statistics/gradients. Active rows must contain the label.
    The vocabulary is equally partitioned, including padded vocabulary IDs,
    exactly as in Megatron's output layer. Only candidate logits cross TP.
    """

    def compute(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: dict[str, Any],
        temperature: float,
        tp_group: dist.ProcessGroup | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Token-set policies require a finite positive temperature")
        support = inputs.get("policy_support")
        if support is None or support.shape[:-1] != labels.shape:
            raise ValueError(
                "Prediction-aligned policy_support is required for every token"
            )
        if logits.shape[:-1] != labels.shape or support.shape[-1] == 0:
            raise ValueError("Logits, labels and candidate metadata are misaligned")
        if support.dtype != torch.long:
            raise ValueError("policy_support must contain int64 token IDs + 1")
        support = support.to(device=logits.device)
        labels = labels.to(device=logits.device)
        valid = support > 0
        active = valid.any(-1)
        ids = support - 1
        rank = dist.get_rank(tp_group) if tp_group is not None else 0
        size = dist.get_world_size(tp_group) if tp_group is not None else 1
        width = logits.shape[-1]
        torch._assert_async(
            torch.all((support >= 0) & (support <= width * size)),
            "Policy support contains an out-of-vocabulary token",
        )
        sorted_support = support.sort(-1).values
        torch._assert_async(
            torch.all(
                (sorted_support[..., 1:] == 0)
                | (sorted_support[..., 1:] != sorted_support[..., :-1])
            ),
            "Policy support contains duplicate tokens",
        )
        local_ids = ids - rank * width
        owned = valid & (local_ids >= 0) & (local_ids < width)
        selected = logits.gather(-1, local_ids.clamp(0, width - 1)).float()
        candidates = torch.where(owned, selected, 0.0)
        if tp_group is not None:
            candidates = _SumCandidateLogits.apply(candidates, tp_group)
        candidates = candidates / temperature
        # Give inactive rows a finite dummy distribution to avoid all--inf softmax.
        fallback = torch.zeros_like(valid)
        fallback[..., 0] = ~active
        finite_support = valid | fallback
        candidates = candidates.masked_fill(~finite_support, -torch.inf)
        logp = torch.log_softmax(candidates, dim=-1)
        match = valid & (ids == labels.unsqueeze(-1))
        torch._assert_async(
            torch.all(~active | match.any(-1)),
            "Sampled label is outside policy support",
        )
        chosen = torch.where(match, logp, 0.0).sum(-1)
        finite_logp = torch.where(finite_support, logp, 0.0)
        entropy = -(finite_logp * logp.exp()).sum(-1)
        return torch.where(active, chosen, 0.0), torch.where(active, entropy, 0.0)


class TokenSetRequest(GenerationRequestPlugin):
    """Use SGLang's public custom-processor API; no backend monkey-patching."""

    @cached_property
    def processor(self) -> str:
        from examples.vlm.pacman.sglang_processor import AllowedTokensProcessor

        return AllowedTokensProcessor.to_str()

    def build_request(self, request: ModelRequest, payload: dict[str, Any]) -> None:
        allowed = request.metadata.get("allowed_token_ids")
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("Every action request must declare allowed_token_ids")
        if any(type(token) is not int or token < 0 for token in allowed) or len(
            set(allowed)
        ) != len(allowed):
            raise ValueError(
                "allowed_token_ids must contain distinct nonnegative integers"
            )
        config = request.gconfig
        if (
            config.greedy
            or config.temperature <= 0
            or config.top_p != 1.0
            or config.max_new_tokens != 1
        ):
            raise ValueError(
                "Released action policies require sampled single-token decoding and top_p=1"
            )
        if (
            config.frequency_penalty != 0
            or config.stop
            or set(config.stop_token_ids).intersection(allowed)
        ):
            raise ValueError(
                "Action policies must not add penalties or early-stop rules"
            )
        payload["custom_logit_processor"] = self.processor
        payload["sampling_params"]["custom_params"] = {"allowed_token_ids": allowed}
        # Disable top-k filtering explicitly, keeping the same support as the learner.
        payload["sampling_params"]["top_k"] = -1
