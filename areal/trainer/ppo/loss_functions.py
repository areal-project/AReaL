# SPDX-License-Identifier: Apache-2.0

"""Built-in token-loss functions, independent of configuration and reduction."""

from collections.abc import Mapping
from typing import Any

import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.trainer.mopd.loss import mopd_loss_fn
from areal.trainer.ppo.policy_loss import (
    _resolve_proximal_logp,
    compute_policy_surrogate,
)
from areal.utils import stats_tracker
from areal.utils.constants import PROX_LOGP_METHOD_RECOMPUTE, ProxLogpMethod
from areal.utils.functional.functional import apply_rejection_sampling


def policy_loss(
    *,
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    mask: torch.Tensor,
    eps_clip: float = 0.2,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    rejection_sampling: RejectionSamplingConfig | None = None,
    importance_sampling_level: str = "token",
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_cispo_loss: bool = False,
    use_decoupled_loss: bool = False,
    advantages_key: str = "advantages",
    advantages: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the PPO/SAPO/CISPO surrogate without reduction or teacher terms."""
    # Teacher objectives are peers in the named interface, never implicit here.
    data = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "teacher_logp",
            "mopd_teacher_logp_sum",
        }
    }
    data["loss_mask"] = mask
    for key in ("logprobs", "prox_logp"):
        if data.get(key) is not None:
            data[key] = torch.where(mask, data[key].detach(), 0.0)
    if advantages is None:
        advantages = data[advantages_key]
    data["advantages"] = torch.where(mask, advantages.detach(), 0.0)
    loss, stat, effective_mask, _ = compute_policy_surrogate(
        torch.where(mask, outputs["logprobs"], 0.0),
        data,
        mask,
        _Unreduced(),
        eps_clip=eps_clip,
        eps_clip_higher=eps_clip_higher,
        c_clip=c_clip,
        rejection_sampling=rejection_sampling,
        importance_sampling_level=importance_sampling_level,
        current_version=batch.get("current_version"),
        prox_logp_method=prox_logp_method,
        use_sapo_loss=use_sapo_loss,
        sapo_tau_pos=sapo_tau_pos,
        sapo_tau_neg=sapo_tau_neg,
        use_cispo_loss=use_cispo_loss,
        use_decoupled_loss=use_decoupled_loss,
    )
    stats_tracker.denominator(n_valid_tokens=effective_mask.bool())
    stats_tracker.stat(
        importance_weight=stat["importance_weight"].float(),
        approx_kl=stat["approx_kl"].float(),
        clip_ratio=stat["clip_mask"].float(),
        dual_clip_ratio=stat["dual_clip_mask"].float(),
        denominator="n_valid_tokens",
    )
    if "behave_mask" in stat:
        stats_tracker.denominator(n_retained_tokens=stat["behave_mask"])
        stats_tracker.stat(
            behave_imp_weight=stat["behave_imp_weight"].float(),
            denominator="n_retained_tokens",
        )
        stats_tracker.scalar(rs_filtered_fraction=stat["filtered_fraction"])
    return loss


class _Unreduced:
    def normalizer(self) -> torch.Tensor:
        raise RuntimeError("Token loss computation does not request a normalizer")

    def aggregate(
        self, loss: torch.Tensor, numerator_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(numerator_mask.bool(), loss, 0.0)


def entropy_loss(
    *,
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Negative entropy: a positive coefficient encourages exploration."""
    if (
        torch.is_grad_enabled()
        and outputs["logprobs"].requires_grad
        and not outputs["entropy"].requires_grad
    ):
        raise ValueError(
            "Entropy loss requires differentiable entropy; enable megatron.entropy_requires_grad for Megatron"
        )
    return -torch.where(mask, outputs["entropy"], 0.0)


def mopd_loss(
    *,
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    mask: torch.Tensor,
    importance_ratio_cap: float = 5.0,
    rejection_sampling: RejectionSamplingConfig | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
) -> torch.Tensor:
    """Multi-teacher distillation with independent reduction and coefficient."""
    behavior_logp = batch.get("mopd_behavior_logprobs")
    if behavior_logp is None:
        raise ValueError("Named MOPD requires immutable mopd_behavior_logprobs")
    normalization_mask = mask
    if rejection_sampling is not None:
        prox = batch.get("prox_logp")
        if ProxLogpMethod(prox_logp_method) == ProxLogpMethod.REUSE_TRAIN_LOGP:
            prox = outputs["logprobs"].detach()
        prox = _resolve_proximal_logp(
            prox_logp_gt=prox,
            prox_logp_method=prox_logp_method,
            old_logp=batch["logprobs"],
            logprobs=outputs["logprobs"].detach(),
            versions=batch.get("versions"),
            current_version=batch.get("current_version"),
        )
        mask = apply_rejection_sampling(
            proximal_logprobs=prox,
            old_logprobs=batch["logprobs"],
            loss_mask=mask,
            cu_seqlens=batch.get("cu_seqlens"),
            config=rejection_sampling,
        ).loss_mask
    loss, _ = mopd_loss_fn(
        logprobs=outputs["logprobs"],
        old_logprobs=behavior_logp,
        teacher_logp_sum=batch["mopd_teacher_logp_sum"],
        teacher_weight_sum=batch["mopd_teacher_weight_sum"],
        loss_mask=mask,
        normalization_mask=normalization_mask,
        importance_ratio_cap=importance_ratio_cap,
        reduction="none",
    )
    return loss


LOSS_FUNCTIONS = {
    "policy": policy_loss,
    "entropy": entropy_loss,
    "mopd": mopd_loss,
}
