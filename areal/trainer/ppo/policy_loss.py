# SPDX-License-Identifier: Apache-2.0

"""Shared policy surrogate and proximal-policy helpers."""

from typing import Any

import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.utils.constants import (
    PROX_APPROX_METHOD_LINEAR,
    PROX_APPROX_METHOD_LOGLINEAR,
    PROX_APPROX_METHOD_ROLLOUT,
    PROX_APPROX_METHODS_ALL,
    PROX_LOGP_METHOD_LOGLINEAR,
    PROX_LOGP_METHOD_METRICS,
    PROX_LOGP_METHOD_RECOMPUTE,
    ProxLogpMethod,
)
from areal.utils.functional import cispo_loss_fn, ppo_actor_loss_fn, sapo_loss_fn
from areal.utils.functional.loss_aggregation import PolicyGradientReduction


def compute_policy_surrogate(
    logprobs: torch.Tensor,
    input_data: dict[str, Any],
    loss_mask: torch.Tensor,
    pg_reduction: PolicyGradientReduction,
    *,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    rejection_sampling: RejectionSamplingConfig | None = None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_cispo_loss: bool = False,
    use_decoupled_loss: bool = False,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor, torch.Tensor | None]:
    """Compute the current main surrogate with the supplied reduction."""
    old_logp = input_data["logprobs"]
    advantages = input_data["advantages"]
    prox_logp_gt = input_data.get("prox_logp")  # Could be None if skipped

    if ProxLogpMethod(prox_logp_method) == ProxLogpMethod.REUSE_TRAIN_LOGP:
        prox_logp_gt = logprobs.detach()

    # Resolve proximal log-probabilities based on method
    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    # Apply M2PO masking if threshold is set
    if m2_threshold is not None:
        loss_mask = _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)

    # Use CISPO, SAPO, or PPO loss
    if use_cispo_loss:
        if use_sapo_loss:
            raise ValueError(
                "CISPO and SAPO are mutually exclusive surrogates. "
                "Set at most one of use_cispo_loss / use_sapo_loss."
            )
        if importance_sampling_level != "token":
            raise ValueError(
                "CISPO only supports importance_sampling_level='token'. "
                "Sequence-level (GSPO-style) CISPO has no published surrogate."
            )
        loss, stat = cispo_loss_fn(
            logprobs=logprobs,
            proximal_logprobs=prox_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            old_logprobs=old_logp,
            rejection_sampling=rejection_sampling,
            cu_seqlens=input_data.get("cu_seqlens"),
            pg_reduction=pg_reduction,
        )
    elif use_sapo_loss:
        if use_decoupled_loss:
            raise ValueError(
                "SAPO is not compatible with `use_decoupled_loss=True`. "
                "Please set `actor.use_decoupled_loss=false` in your configuration."
            )
        loss, stat = sapo_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            tau_pos=sapo_tau_pos,
            tau_neg=sapo_tau_neg,
            loss_mask=loss_mask,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
            pg_reduction=pg_reduction,
        )
    else:
        loss, stat = ppo_actor_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logp,
            advantages=advantages,
            eps_clip=eps_clip,
            eps_clip_higher=eps_clip_higher,
            loss_mask=loss_mask,
            c_clip=c_clip,
            proximal_logprobs=prox_logp,
            rejection_sampling=rejection_sampling,
            importance_sampling_level=importance_sampling_level,
            cu_seqlens=input_data.get("cu_seqlens"),
            pg_reduction=pg_reduction,
        )

    return loss, stat, loss_mask, prox_logp_gt


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Compute approximation(s) for proximal policy log-probabilities.

    This function approximates the log-probabilities of the proximal policy (one training step
    behind the current policy) using version-aware interpolation between the behavior policy
    (old_logp) and current policy (logprobs). This avoids the need for an expensive forward pass
    to compute the proximal policy's log-probabilities explicitly.

    Args:
        old_logp: log_p_behave from the rollout (behavior policy)
        logprobs: log_p_theta from current training forward pass
        versions: per-token policy versions from rollout (v_behave for each token)
        current_version: current training step version (v_theta)
        method: If specified, only compute this method. If None, compute all methods.

    Returns:
        Dictionary with approximation results. Single key if method specified, all methods otherwise.
    """
    # Assume proximal version is current_version - 1 (last broadcast)
    # In AReaL, proximal policy is the last updated/broadcast policy version
    v_proximal = current_version - 1

    # Extract version information
    v_behave = versions.float()
    v_theta = float(current_version)

    # CRITICAL: Only approximate generated tokens (version >= 0)
    # Prompt tokens (version < 0) must NOT be approximated - they have no generation version
    generated_tokens_mask = versions >= 0

    # Compute interpolation factor alpha
    # When v_behave == v_proximal: alpha=0 (use old_logp)
    # When v_behave == v_theta: alpha=1 (use logprobs)
    # For prompt tokens (version < 0): alpha=0 (no interpolation)
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    # Avoid division by zero AND exclude prompt tokens
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask,
        version_gap / version_diff,
        torch.zeros_like(v_behave),
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)

    approximations = {}

    # If method is specified, only compute that one
    # Otherwise compute all methods (for metrics comparison)
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL

    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            # Method 1: Log-linear interpolation in log-space (geometric mean in probability space)
            # log(p_prox) = (1-α)·log(p_behave) + α·log(p_theta)
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (
                logprobs - old_logp
            )

        elif m == PROX_APPROX_METHOD_LINEAR:
            # Method 2: Linear interpolation in probability space (arithmetic mean)
            # p_prox = (1-α)·p_behave + α·p_theta
            # Then convert back to log space: log(p_prox)
            p_behave = torch.exp(old_logp)
            p_theta = torch.exp(logprobs)
            p_arithmetic = (1 - alpha) * p_behave + alpha * p_theta
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)

        elif m == PROX_APPROX_METHOD_ROLLOUT:
            # Method 3: Use behavior policy from rollout as-is (no approximation)
            # p_prox = p_behave
            # Used for metrics comparison
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()

    return approximations


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor:
    """
    Resolve the proximal policy log-probabilities based on the method.

    This function determines the final proximal log-probabilities to use for PPO training,
    either from ground truth (forward pass) or approximation methods.

    Args:
        prox_logp_gt: Ground truth proximal logp (from forward pass), or None if skipped.
        prox_logp_method: Method to use (recompute, loglinear, metrics).
        old_logp: Behavior policy log-probabilities.
        logprobs: Current policy log-probabilities (should be detached).
        versions: Per-token policy versions, or None.
        current_version: Current training version, or None.

    Returns:
        Resolved proximal log-probabilities tensor.

    Raises:
        ValueError: If configuration is invalid (e.g., missing required data).
        RuntimeError: If computation fails (None result, NaN, Inf).
    """
    prox_logp_is_none = prox_logp_gt is None

    # Validate configuration when prox_logp is None
    if prox_logp_is_none:
        if not ProxLogpMethod(prox_logp_method).skips_forward_pass():
            raise ValueError(
                f"prox_logp is None but prox_logp_method='{prox_logp_method}'. "
                "This indicates compute_logp() was skipped incorrectly."
            )
        if versions is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' "
                "but versions not available. "
                "Cannot proceed without either ground truth or approximation."
            )

    # Determine prox_logp based on method
    prox_logp = prox_logp_gt  # Default to ground truth (could be None)

    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        # Use loglinear approximation (must compute if prox_logp is None)
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    elif prox_logp_method == PROX_LOGP_METHOD_METRICS:
        # Metrics mode: use recomputed prox_logp for training,
        # but will also compute approximation metrics later
        pass  # Use prox_logp_gt as-is (should be recomputed)
    # else: PROX_LOGP_METHOD_RECOMPUTE - use prox_logp_gt as-is

    # Safety check: ensure we have prox_logp
    if prox_logp is None:
        raise RuntimeError(
            f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'. "
            "This indicates configuration or computation error."
        )

    # Verify the value is valid
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(
            f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'. "
            "This indicates computation failed."
        )

    return prox_logp


def _apply_m2po_masking(
    old_logp: torch.Tensor,
    prox_logp: torch.Tensor,
    loss_mask: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Apply M2PO (Second-Momentum PPO) masking to filter high-variance tokens.

    M2PO filters out tokens with high second-momentum (squared difference between
    old and proximal log-probabilities) to reduce gradient variance.

    Args:
        old_logp: Behavior policy log-probabilities.
        prox_logp: Proximal policy log-probabilities.
        loss_mask: Original loss mask [batch, seq_len].
        m2_threshold: Threshold for second-momentum filtering.

    Returns:
        Updated loss mask with M2PO filtering applied.
    """
    delta = old_logp - prox_logp
    m2 = delta * delta
    mask_flat = loss_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]

    if m2_selected.numel() == 0:
        return loss_mask

    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)
    sorted_m2_loss_mask = _get_m2po_loss_mask(
        sorted_m2=sorted_m2, m2_threshold=m2_threshold
    )
    m2_selected_mask = sorted_m2_loss_mask[restored_indices]

    m2_full_flat = torch.zeros_like(
        mask_flat, dtype=torch.bool, device=loss_mask.device
    )
    m2_full_flat[mask_flat] = m2_selected_mask

    return m2_full_flat.view_as(loss_mask)


def _get_m2po_loss_mask(
    sorted_m2: torch.Tensor,
    m2_threshold: float,
) -> torch.Tensor:
    """
    Get the mask for M2PO loss based on the second-momentum threshold.
    Mask the tokens whose second-momentum is the largest, until the average second-momentum is below the threshold.
    """
    n = sorted_m2.numel()
    if n == 0:
        return torch.ones_like(sorted_m2, dtype=torch.bool)

    # Suffix sums: S[i] = sum(sorted_m2[i:])
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)

    # Number of elements in suffix: N[i] = n - i
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)

    # Average of suffix: A[i] = S[i] / N[i]
    avg_m2_suffix = suffix_sums / counts

    # Find the first index `k` where the average of the rest is below threshold.
    below_threshold_indices = torch.where(avg_m2_suffix < m2_threshold)[0]

    if len(below_threshold_indices) > 0:
        num_to_mask = below_threshold_indices[0].item()
    else:
        # All suffix averages are >= threshold. Mask all but one to satisfy assertion.
        num_to_mask = n - 1

    loss_mask = torch.ones_like(sorted_m2, dtype=torch.bool)
    if num_to_mask > 0:
        loss_mask[:num_to_mask] = False

    if loss_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when getting the m2po loss mask.")

    return loss_mask
