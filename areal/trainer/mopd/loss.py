# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch

from areal.api.cli_args import MOPDLossConfig

DEFAULT_MOPD_IMPORTANCE_RATIO_CAP = 5.0


def _validate_mopd_tensor_shapes(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    teacher_logp_sum: torch.Tensor,
    teacher_weight_sum: torch.Tensor,
    loss_mask: torch.Tensor,
) -> None:
    expected_shape = logprobs.shape
    tensors = {
        "old_logprobs": old_logprobs,
        "teacher_logp_sum": teacher_logp_sum,
        "teacher_weight_sum": teacher_weight_sum,
        "loss_mask": loss_mask,
    }
    for name, tensor in tensors.items():
        if tensor.shape != expected_shape:
            raise ValueError(
                f"{name} must have token shape {expected_shape}, got {tensor.shape}"
            )


def mopd_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    teacher_logp_sum: torch.Tensor,
    teacher_weight_sum: torch.Tensor,
    loss_mask: torch.Tensor,
    normalization_mask: torch.Tensor | None = None,
    importance_ratio_cap: float = DEFAULT_MOPD_IMPORTANCE_RATIO_CAP,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a truncated-IS multi-teacher reverse-KL surrogate.

    ``teacher_logp_sum`` is ``sum_j(w_j * log pi_Tj)`` and
    ``teacher_weight_sum`` is ``sum_j(w_j)`` at every response token. Teacher
    weights are deliberately not normalized. The behavior-policy importance
    ratio is capped with a stop-gradient weight so extreme stale tokens retain
    a bounded policy gradient instead of overflowing or becoming constant.
    """
    if (
        not isinstance(importance_ratio_cap, (int, float))
        or isinstance(importance_ratio_cap, bool)
        or not math.isfinite(importance_ratio_cap)
        or importance_ratio_cap <= 0
    ):
        raise ValueError("importance_ratio_cap must be a finite positive number")
    _validate_mopd_tensor_shapes(
        logprobs,
        old_logprobs,
        teacher_logp_sum,
        teacher_weight_sum,
        loss_mask,
    )
    if normalization_mask is not None and normalization_mask.shape != logprobs.shape:
        raise ValueError(
            "normalization_mask must have token shape "
            f"{logprobs.shape}, got {normalization_mask.shape}"
        )

    mask = loss_mask.bool()
    safe_logprobs = torch.where(mask, logprobs, torch.zeros_like(logprobs))
    detached_teacher_logp = torch.where(
        mask, teacher_logp_sum.detach(), torch.zeros_like(teacher_logp_sum)
    )
    detached_teacher_weight = torch.where(
        mask, teacher_weight_sum.detach(), torch.zeros_like(teacher_weight_sum)
    )
    detached_old_logprobs = torch.where(
        mask, old_logprobs.detach(), torch.zeros_like(old_logprobs)
    )
    active_inputs_are_finite = (
        torch.isfinite(safe_logprobs)
        & torch.isfinite(detached_teacher_logp)
        & torch.isfinite(detached_teacher_weight)
        & torch.isfinite(detached_old_logprobs)
    ).all()
    torch._assert_async(
        active_inputs_are_finite,
        "MOPD loss inputs must be finite at active tokens",
    )
    # Sanitize masked positions before nonlinear operations. Applying the mask
    # only after exp() can leave an infinite intermediate whose backward is
    # 0 * inf = NaN, even though that token contributes zero to the loss.
    detached_log_ratio = safe_logprobs.detach() - detached_old_logprobs
    bounded_importance_weight = torch.exp(
        detached_log_ratio.clamp(max=math.log(importance_ratio_cap))
    )
    score_reward = torch.where(
        mask,
        detached_teacher_logp - (detached_teacher_weight * safe_logprobs.detach()),
        torch.zeros_like(logprobs),
    )
    # At forward time the carrier is exactly one. Its derivative is
    # d log pi_theta, preserving a bounded score-function gradient even when
    # the detached importance ratio was clipped.
    score_function_carrier = torch.exp(safe_logprobs - safe_logprobs.detach())
    importance_weight = bounded_importance_weight * score_function_carrier
    per_token_loss = -(importance_weight * score_reward)
    masked_per_token_loss = per_token_loss
    denominator_mask = mask if normalization_mask is None else normalization_mask.bool()
    denominator = denominator_mask.count_nonzero().clamp_min(1)
    loss = masked_per_token_loss.sum() / denominator

    reverse_kl = bounded_importance_weight * (-score_reward)
    stats = {
        "loss": loss.detach(),
        "loss_per_token": masked_per_token_loss.detach(),
        "score_reward": torch.where(mask, score_reward, torch.zeros_like(score_reward)),
        "importance_weight": torch.where(
            mask, importance_weight.detach(), torch.zeros_like(importance_weight)
        ),
        "teacher_weight_sum": torch.where(
            mask,
            detached_teacher_weight,
            torch.zeros_like(detached_teacher_weight),
        ),
        "reverse_kl": torch.where(mask, reverse_kl, torch.zeros_like(reverse_kl)),
    }
    return loss, stats


def compose_mopd_loss(
    rl_loss: torch.Tensor,
    *,
    config: MOPDLossConfig | None,
    logprobs: torch.Tensor | None = None,
    old_logprobs: torch.Tensor | None = None,
    teacher_logp_sum: torch.Tensor | None = None,
    teacher_weight_sum: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    normalization_mask: torch.Tensor | None = None,
    importance_ratio_cap: float = DEFAULT_MOPD_IMPORTANCE_RATIO_CAP,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compose RL and MOPD objectives without changing the disabled RL path."""
    if config is None:
        return rl_loss, {}

    required_tensors = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "teacher_logp_sum": teacher_logp_sum,
        "teacher_weight_sum": teacher_weight_sum,
        "loss_mask": loss_mask,
    }
    missing = [name for name, tensor in required_tensors.items() if tensor is None]
    if missing:
        raise ValueError(f"MOPD loss inputs are missing: {', '.join(missing)}")

    assert logprobs is not None
    assert old_logprobs is not None
    assert teacher_logp_sum is not None
    assert teacher_weight_sum is not None
    assert loss_mask is not None
    mopd_loss, stats = mopd_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        teacher_logp_sum=teacher_logp_sum,
        teacher_weight_sum=teacher_weight_sum,
        loss_mask=loss_mask,
        normalization_mask=normalization_mask,
        importance_ratio_cap=importance_ratio_cap,
    )

    if config.rl_coefficient == 0:
        total_loss = config.distillation_coefficient * mopd_loss
    elif config.distillation_coefficient == 0:
        total_loss = config.rl_coefficient * rl_loss
    else:
        total_loss = (
            config.rl_coefficient * rl_loss
            + config.distillation_coefficient * mopd_loss
        )

    stats["total_loss"] = total_loss.detach()
    return total_loss, stats
