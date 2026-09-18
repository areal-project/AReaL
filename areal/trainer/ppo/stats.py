# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Any

import torch

from areal.utils import stats_tracker
from areal.utils.stats_tracker import ReduceType


def log_train_inference_stats(
    trainer_logp: torch.Tensor,
    rollout_logp: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Compare recomputed and original rollout logprobs before PPO overwrites them.

    Both inputs use next-token alignment. This includes policy drift for stale
    rollouts; it measures engine agreement only for matching policy versions.
    Ratio tails follow R3's extreme token fraction F(tau), Eq. (3) in
    https://arxiv.org/abs/2510.11370. Matching a live dashboard also requires
    matching its mask, sampling convention, and compared policy versions.
    """
    if trainer_logp.shape != rollout_logp.shape or trainer_logp.shape != mask.shape:
        raise ValueError("Trainer logprobs, rollout logprobs, and mask must match")
    with stats_tracker.scope("ppo_actor/train_infer"):
        mask = mask.bool()
        trainer_logp = trainer_logp.detach().float()
        rollout_logp = rollout_logp.detach().float()
        finite_logps = torch.isfinite(trainer_logp) & torch.isfinite(rollout_logp)
        delta = torch.where(mask, trainer_logp - rollout_logp, 0.0)
        stats_tracker.denominator(n_valid_tokens=mask)
        stats_tracker.stat(
            logp_diff=delta,
            logp_abs_diff=delta.abs(),
            denominator="n_valid_tokens",
        )
        stats_tracker.stat(
            trainer_nll=torch.where(mask, -trainer_logp, 0.0),
            rollout_nll=torch.where(mask, -rollout_logp, 0.0),
            kl_k1=-delta,
            kl_k2=delta.square() / 2,
            kl_k3=delta.expm1() - delta,
            logp_diff_squared=delta.square(),
            nonfinite_logp_fraction=(~finite_logps).float(),
            denominator="n_valid_tokens",
            reduce_type=ReduceType.AVG,
        )
        for threshold in (1.5, 2, 3, 5, 10):
            stats_tracker.stat(
                **{
                    # A NaN comparison is false, which would otherwise turn
                    # corrupt valid tokens into apparently healthy zero tails.
                    f"ratio_outside_{threshold}": torch.where(
                        finite_logps,
                        (delta.abs() > math.log(threshold)).float(),
                        float("nan"),
                    )
                },
                denominator="n_valid_tokens",
                reduce_type=ReduceType.AVG,
            )


def infer_token_denominator(
    input_data: dict[str, Any],
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Infer the full token mask for stats logging.

    Context parallelism may slice intermediate tensors such as ``loss_mask`` or
    model outputs, while the original micro-batch metadata still describes the
    full logical sequence. Prefer that metadata for ``n_tokens`` so statistics
    stay consistent with and without context parallelism.
    """
    common_kwargs = {"dtype": torch.bool, "device": fallback.device}

    attention_mask = input_data.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return torch.ones_like(attention_mask, **common_kwargs)

    cu_seqlens = input_data.get("cu_seqlens")
    if isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.numel() > 0:
        return torch.ones(int(cu_seqlens[-1].item()), **common_kwargs)

    input_ids = input_data.get("input_ids")
    # Tree-packed batches keep input_ids padded to tree size while token-level
    # stats stay at packed-token length. Only reuse input_ids when it already
    # matches the stat tensor shape.
    if isinstance(input_ids, torch.Tensor) and input_ids.shape == fallback.shape:
        return torch.ones_like(input_ids, **common_kwargs)

    return torch.ones_like(fallback, **common_kwargs)
