# SPDX-License-Identifier: Apache-2.0
"""
Fused linear cross-entropy entry points for AReaL.

These wrappers bridge the fused Triton kernel into AReaL's
:func:`gather_logprobs_entropy` interface so the Megatron path can opt in
via a single config flag.  They fall back to the materialised reference
path when Triton is unavailable or inputs are not on CUDA.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from areal.utils import logging

logger = logging.getLogger("LinearCrossEntropy")


def _force_fallback() -> bool:
    return os.environ.get("AREAL_DISABLE_FUSED_LCE", "0") == "1"


def _kernel_available() -> bool:
    if _force_fallback():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _reference_logprobs_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    tp_group: dist.ProcessGroup | None,
    return_max_logits: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_labels = labels.reshape(-1)

    logits = torch.matmul(flat_hidden.float(), weight.float().t())
    if temperature != 1.0:
        logits = logits / temperature

    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        world_size = dist.get_world_size(tp_group)
        gathered = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(gathered, logits, group=tp_group)
        logits = torch.cat(gathered, dim=-1)

    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_softmax.gather(
        dim=-1, index=flat_labels.unsqueeze(-1)
    ).squeeze(-1)
    probs = log_softmax.exp()
    entropy = -(probs * log_softmax).sum(dim=-1)
    if return_max_logits:
        # Return max of the post-temperature logits, scaled back by ``temperature``
        # so the value matches ``raw_logits.max(-1).values`` (matches the
        # non-fused telemetry path exactly).
        max_logits = logits.detach().max(dim=-1).values.float()
        if temperature != 1.0:
            max_logits = max_logits * temperature
        return log_probs_labels, entropy, max_logits
    return log_probs_labels, entropy


def linear_cross_entropy_logprobs_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
    return_max_logits: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Compute per-token log-prob and entropy via the fused kernel.

    Falls back to the materialised reference path when the fused kernel is
    unavailable.

    Args:
        hidden: ``(..., hidden_size)`` last-layer hidden states.
        weight: ``(vocab_size, hidden_size)`` lm-head weight; may be
            vocab-sharded when ``tp_group`` is set.
        labels: ``(...,)`` integer label ids. With TP, labels must hold
            *global* vocab ids.
        temperature: softmax temperature.
        tp_group: optional tensor-parallel group when ``weight`` is sharded.
        return_max_logits: when ``True``, additionally returns the per-token
            max of the **raw** (pre-temperature) logits, shape ``labels.shape``,
            dtype ``float32``. The fused kernel internally tracks
            ``max(logits/temperature)``; we multiply it back by ``temperature``
            so the value is numerically identical to
            ``raw_logits.max(-1).values`` from the non-fused path.

    Returns:
        ``(logprobs, entropy)`` both shaped like ``labels``; or
        ``(logprobs, entropy, max_logits)`` when ``return_max_logits=True``.
    """
    leading_shape = labels.shape

    if _kernel_available():
        from areal.models.kernel.linear_cross_entropy import linear_cross_entropy

        if hidden.device.type != "cuda":
            logger.warning(
                "Fused LCE requested but hidden is on %s; falling back to reference.",
                hidden.device,
            )
        else:
            try:
                if return_max_logits:
                    logprobs, entropy, max_logits = linear_cross_entropy(
                        hidden,
                        weight,
                        labels,
                        temperature,
                        "none",
                        tp_group,
                        return_max_logits=True,
                    )
                    return (
                        logprobs.reshape(leading_shape),
                        entropy.reshape(leading_shape),
                        max_logits.reshape(leading_shape),
                    )
                logprobs, entropy = linear_cross_entropy(
                    hidden,
                    weight,
                    labels,
                    temperature,
                    "none",
                    tp_group,
                )
                return logprobs.reshape(leading_shape), entropy.reshape(leading_shape)
            except Exception as exc:
                logger.warning(
                    "Fused LCE kernel raised %s; falling back to reference.",
                    exc,
                )

    if return_max_logits:
        logprobs, entropy, max_logits = _reference_logprobs_entropy(
            hidden,
            weight,
            labels,
            temperature,
            tp_group,
            return_max_logits=True,
        )
        return (
            logprobs.reshape(leading_shape),
            entropy.reshape(leading_shape),
            max_logits.reshape(leading_shape),
        )
    logprobs, entropy = _reference_logprobs_entropy(
        hidden, weight, labels, temperature, tp_group
    )
    return logprobs.reshape(leading_shape), entropy.reshape(leading_shape)


def linear_cross_entropy_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    tp_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Logprobs-only counterpart of :func:`linear_cross_entropy_logprobs_entropy`."""
    logprobs, _ = linear_cross_entropy_logprobs_entropy(
        hidden, weight, labels, temperature, tp_group
    )
    return logprobs
