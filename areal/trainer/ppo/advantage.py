# SPDX-License-Identifier: Apache-2.0

"""Composable PPO advantage computation with worker-local callable resolution."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import torch

from areal.api.cli_args import AdvantageFunctionConfig, PPOActorConfig
from areal.trainer.ppo.advantage_algorithms import (
    ProcessAdvantageInputs,
    TemporalInputs,
    TemporalResult,
    additive,
    gvpo,
    process_weighted,
    token_gae,
    turn_gae,
)
from areal.trainer.ppo.lambda_fn import resolve_gae_lambda_fn
from areal.trainer.ppo.stats import log_train_inference_stats
from areal.utils.data import (
    KLEstimator,
    Normalization,
    TrajBatchMeta,
    normalize_rollout_rewards,
)
from areal.utils.dynamic_import import import_from_string
from areal.utils.functional import reward_overlong_penalty


@dataclass(frozen=True)
class AdvantageInputs:
    """Rollout tensors and immutable grouping metadata; row order must be preserved."""

    data: Mapping[str, Any]
    config: PPOActorConfig
    meta: TrajBatchMeta | None = None
    process_implementation: str = "additive"
    process_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutcomeRewardInputs:
    """Raw rollout fields after length penalty and before scaling/normalization.

    Plugins must preserve row order and logical rollout membership. Metadata
    carries raw logical reward references, not shaped normalization references.
    """

    data: Mapping[str, Any]
    config: PPOActorConfig
    meta: TrajBatchMeta | None = None


@dataclass(frozen=True)
class OutcomeResult:
    rewards: torch.Tensor
    normalization_rewards: torch.Tensor | None = None


@dataclass(frozen=True)
class AdvantageResult:
    advantages: torch.Tensor
    returns: torch.Tensor
    loss_mask: torch.Tensor
    logprobs: torch.Tensor
    rewards: torch.Tensor
    total_rewards: torch.Tensor
    kl_rewards: torch.Tensor
    extras: Mapping[str, torch.Tensor] = field(default_factory=dict)


DERIVED_FIELDS = frozenset(
    {
        "input_token_loss_mask",
        "is_truncated",
        "mopd_behavior_logprobs",
        "prox_logp",
        "ref_logp",
    }
)
PROCESS_COMPOSERS = {
    "additive": additive,
    "gvpo": gvpo,
    "process_weighted": process_weighted,
}


def identity_outcome(inputs: OutcomeRewardInputs) -> OutcomeResult:
    return OutcomeResult(rewards=inputs.data["rewards"])


def _resolve_function(
    spec: AdvantageFunctionConfig, registry: Mapping[str, Callable], stage: str
) -> partial:
    fn = registry.get(spec.implementation)
    if fn is None:
        if "." not in spec.implementation:
            raise ValueError(f"Unknown {stage} implementation: {spec.implementation!r}")
        fn = import_from_string(spec.implementation)
    if not callable(fn):
        raise TypeError(
            f"{stage} implementation must be callable: {spec.implementation}"
        )
    return partial(fn, **spec.kwargs)


def _validate_tensor(
    value: torch.Tensor,
    reference: torch.Tensor,
    name: str,
    *,
    boolean: bool = False,
    original_reward: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.shape != reference.shape or value.device != reference.device:
        raise ValueError(
            f"{name} must match shape {reference.shape} and device {reference.device}"
        )
    if not torch.is_floating_point(value) and not (
        (boolean and value.dtype == torch.bool)
        or (
            original_reward
            and value.dtype == reference.dtype
            and not value.is_complex()
        )
    ):
        raise TypeError(
            f"{name} must have a floating-point dtype" + (" or bool" if boolean else "")
        )
    torch._assert_async(torch.all(torch.isfinite(value)), f"{name} must be finite")


def validate_advantage_result(
    result: AdvantageResult, data: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    if not isinstance(result, AdvantageResult):
        raise TypeError("advantage compute must return AdvantageResult")
    output = {}
    for name in (
        "advantages",
        "returns",
        "loss_mask",
        "logprobs",
        "rewards",
        "total_rewards",
        "kl_rewards",
    ):
        value = getattr(result, name)
        reference = data["rewards"] if name == "rewards" else data["loss_mask"]
        _validate_tensor(
            value,
            reference,
            name,
            boolean=name == "loss_mask",
            original_reward=name == "rewards",
        )
        if name == "loss_mask":
            torch._assert_async(
                torch.all((value == 0) | (value == 1)), "loss_mask must be binary"
            )
        output["tot_rewards" if name == "total_rewards" else name] = value
    if not isinstance(result.extras, Mapping):
        raise TypeError("advantage extras must be a mapping")
    for name, value in result.extras.items():
        if name not in DERIVED_FIELDS:
            raise ValueError(f"Unsupported advantage extra field: {name!r}")
        reference = data["rewards"] if name == "is_truncated" else data["loss_mask"]
        _validate_tensor(
            value,
            reference,
            name,
            boolean=name in {"is_truncated", "input_token_loss_mask"},
        )
        if name in {"is_truncated", "input_token_loss_mask"}:
            torch._assert_async(
                torch.all((value == 0) | (value == 1)), f"{name} must be binary"
            )
        output[name] = value
    return output


def resolve_advantage_fn(
    config: PPOActorConfig,
) -> Callable[[AdvantageInputs], AdvantageResult]:
    options = config.advantage
    # Resolve aliases before checking exclusivity, just as for individual stages.
    selected = _resolve_function(
        options.compute, {"default": compute_advantages}, "compute"
    )
    if selected.func is compute_advantages:
        if selected.keywords:
            raise ValueError(
                "default advantage compute does not accept kwargs; use stage kwargs"
            )
        return AdvantagePipeline(config)
    if any(
        getattr(options, name) is not None
        for name in ("outcome_reward_shaping", "estimator", "process_adv_shaping")
    ):
        raise ValueError(
            "a custom advantage compute cannot be combined with stage overrides"
        )
    return selected


def compute_advantages(inputs: AdvantageInputs) -> AdvantageResult:
    """Standalone entry point; actors resolve and cache the pipeline once."""
    return AdvantagePipeline(inputs.config)(inputs)


def _get_truncated_mask(data: dict[str, Any], seqlens: torch.Tensor) -> torch.Tensor:
    is_truncated = data.get("is_truncated")
    if is_truncated is None:
        # Preserve compatibility with custom tensor workflows that predate the
        # explicit per-trajectory termination metadata.
        return seqlens == data["attention_mask"].shape[-1]
    if not torch.is_tensor(is_truncated):
        raise TypeError("`is_truncated` must be a tensor")
    if is_truncated.shape != seqlens.shape:
        raise ValueError(
            "`is_truncated` must have one value per trajectory, got "
            f"shape {tuple(is_truncated.shape)} for batch shape {tuple(seqlens.shape)}"
        )
    return is_truncated.to(device=seqlens.device, dtype=torch.bool)


class AdvantagePipeline:
    """Default pipeline; resolved callables and normalizers are local to each worker."""

    def __init__(self, config: PPOActorConfig):
        self.config = config
        self.reward_bias = config.reward_bias
        self.reward_scaling = config.reward_scaling
        self.reward_clip = config.reward_clip

        self.kl_ctl = config.kl_ctl
        self.kl_estimator = KLEstimator(config.kl_estimator)

        self.adv_norm = Normalization(config.adv_norm) if config.adv_norm else None
        self.reward_norm = (
            Normalization(config.reward_norm) if config.reward_norm else None
        )

        self.discount = config.discount
        self.gae_lambda = config.gae_lambda
        self.gae_lambda_fn, self._gae_lambda_is_custom = resolve_gae_lambda_fn(
            config.gae_lambda
        )
        self.gae_lambda_kwargs = (
            dict(config.gae_lambda_kwargs) if self._gae_lambda_is_custom else {}
        )
        self.gae_timestep_unit = config.gae_timestep_unit
        self.mask_no_eos_with_zero = config.mask_no_eos_with_zero
        self.token_rewards_as_adv = config.token_rewards_as_adv

        options = config.advantage

        def resolve(spec, default, registry, stage):
            if spec is None or spec.implementation == "auto":
                spec = AdvantageFunctionConfig(
                    implementation=default, kwargs={} if spec is None else spec.kwargs
                )
            return _resolve_function(spec, registry, stage)

        self.outcome = resolve(
            options.outcome_reward_shaping,
            "identity",
            {"identity": identity_outcome},
            "outcome_reward_shaping",
        )
        self.estimator = resolve(
            options.estimator,
            f"{config.gae_timestep_unit}_gae",
            {"token_gae": token_gae, "turn_gae": turn_gae},
            "estimator",
        )
        process = options.process_adv_shaping
        if process is not None and process.implementation == "auto":
            if process.kwargs:
                raise ValueError("process_adv_shaping auto mode does not accept kwargs")
            process = None
        self.process = (
            None
            if process is None
            else _resolve_function(process, PROCESS_COMPOSERS, "process_adv_shaping")
        )

    def __call__(self, inputs: AdvantageInputs) -> AdvantageResult:
        data = dict(inputs.data)
        meta = inputs.meta
        process = self.process or _resolve_function(
            AdvantageFunctionConfig(implementation=inputs.process_implementation),
            PROCESS_COMPOSERS,
            "process_adv_shaping",
        )
        process_kwargs = dict(process.keywords)
        if process.func is gvpo:
            process_kwargs = {**inputs.process_kwargs, **process_kwargs}
        bs = data["input_ids"].shape[0]
        batch_indices = torch.arange(
            bs, device=data["input_ids"].device, dtype=torch.long
        )

        unpenalized_rewards = None
        # Reward Penalty on length
        if self.config.overlong_reward_penalty:
            if (
                self.reward_norm
                and meta is not None
                and meta.rollout_groups is not None
            ):
                unpenalized_rewards = data["rewards"].clone()
            overlong_tokens = self.config.overlong_tokens
            overlong_penalty_factor = self.config.overlong_penalty_factor

            assert overlong_tokens is not None
            assert overlong_penalty_factor is not None
            data = reward_overlong_penalty(
                data,
                overlong_tokens=overlong_tokens,
                overlong_penalty_factor=overlong_penalty_factor,
                max_response_length=self.config.max_new_tokens,
            )

        outcome = self.outcome(
            OutcomeRewardInputs(data=data, config=self.config, meta=meta)
        )
        if not isinstance(outcome, OutcomeResult):
            raise TypeError("outcome_reward_shaping must return OutcomeResult")
        _validate_tensor(
            outcome.rewards,
            data["rewards"],
            "outcome rewards",
            original_reward=self.outcome.func is identity_outcome,
        )
        reward_score = outcome.rewards
        if outcome.normalization_rewards is not None and (
            meta is None or meta.rollout_groups is None
        ):
            raise ValueError(
                "normalization_rewards requires complete trajectory metadata from concat_batch"
            )
        # Reward Scaling
        reward_score = (reward_score + self.reward_bias) * self.reward_scaling
        reward_score = torch.clip(
            reward_score, max=self.reward_clip, min=-self.reward_clip
        )
        # Use actual trajectory group sizes when available so group-level
        # normalization handles failed/filtered rollout samples without slicing
        # across prompts. Direct calls without batched metadata keep the legacy
        # fixed-group-size behavior.
        group_sizes = meta.traj_group_sizes if meta is not None else None
        if self.reward_norm:
            if meta is not None and meta.rollout_groups is not None:
                reward_score = normalize_rollout_rewards(
                    outcome.rewards,
                    self.reward_norm,
                    meta,
                    reward_bias=self.reward_bias,
                    reward_scaling=self.reward_scaling,
                    reward_clip=self.reward_clip,
                    unpenalized_rewards=unpenalized_rewards,
                    normalization_rewards=outcome.normalization_rewards,
                    require_shaped_references=self.outcome.func is not identity_outcome,
                )
            else:
                reward_score = self.reward_norm(reward_score, group_sizes=group_sizes)

        token_loss_mask = data["loss_mask"].bool()
        # Preserve token provenance before converting to next-token loss alignment.
        # Multimodal preprocessing must distinguish generated special IDs from
        # prompt image/video placeholders even after advantages are computed.
        data["input_token_loss_mask"] = token_loss_mask
        loss_mask = token_loss_mask.float()
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=-1)

        if "mopd_teacher_logp_sum" in data:
            # MOPD's correction is always relative to the immutable rollout
            # behavior policy, even when standard PPO recomputes its proximal
            # policy and overwrites ``logprobs`` below.
            data["mopd_behavior_logprobs"] = (
                torch.roll(data["logprobs"], shifts=-1, dims=-1) * loss_mask
            ).detach()

        # Align structural turn IDs to the same next-token prediction
        # convention used by loss_mask and log probabilities.
        turn_ids = data.get("turn_ids")
        if turn_ids is not None:
            turn_ids = torch.roll(turn_ids, shifts=-1, dims=-1)
            turn_ids[:, -1] = -1
        # Apply the mask to log probabilities.
        if data.get("prox_logp") is not None:
            log_train_inference_stats(
                data["prox_logp"],
                torch.roll(data["logprobs"], shifts=-1, dims=-1),
                loss_mask.bool(),
            )
        if not self.config.use_decoupled_loss and self.config.recompute_logprob:
            # Overwrite logprobs produced by the inference engine
            prox_logp_value = data["prox_logp"]
            if prox_logp_value is None:
                raise ValueError(
                    "prox_logp is None but recompute_logprob=True. "
                    "This indicates compute_logp() was skipped incorrectly."
                )
            old_logp = data["logprobs"] = prox_logp_value
        else:
            old_logp = torch.roll(data["logprobs"], shifts=-1, dims=-1)
            if not self.config.use_decoupled_loss:
                # prox logp not available, use inferenced logp
                data["prox_logp"] = old_logp
        ref_logp = data.get("ref_logp")
        if ref_logp is None:
            ref_logp = torch.zeros_like(old_logp)
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute KL-regularized rewards.
        attn_mask = data["attention_mask"]
        seqlens = attn_mask.sum(-1).long()
        seq_truncated_mask = _get_truncated_mask(data, seqlens)
        data["is_truncated"] = seq_truncated_mask
        rewards = -self.kl_ctl * self.kl_estimator(old_logp, ref_logp)
        kl_rewards = rewards.clone()
        # KL rewards at the next token after eos is zero.
        rewards[batch_indices, seqlens - 1] = 0
        gae_kl_rewards = rewards.clone()
        indices = torch.clip(seqlens - 2, min=0)
        gae_outcome_rewards = torch.zeros_like(rewards)
        if self.mask_no_eos_with_zero:
            gae_outcome_rewards[batch_indices, indices] = torch.where(
                seq_truncated_mask, 0, reward_score
            )
        else:
            gae_outcome_rewards[batch_indices, indices] = reward_score

        process_rewards = None
        token_advantages = None
        token_rewards = data.get("token_rewards")
        if token_rewards is not None:
            if token_rewards.shape != rewards.shape:
                raise ValueError(
                    "token_rewards must match the padded sequence shape: "
                    f"expected {tuple(rewards.shape)}, got {tuple(token_rewards.shape)}"
                )
            aligned_token_rewards = token_rewards.to(
                device=rewards.device, dtype=rewards.dtype
            )
            if self.mask_no_eos_with_zero:
                aligned_token_rewards = torch.where(
                    seq_truncated_mask.unsqueeze(-1),
                    torch.zeros_like(aligned_token_rewards),
                    aligned_token_rewards,
                )
            rolled_token_rewards = torch.roll(
                aligned_token_rewards,
                shifts=-1,
                dims=-1,
            )
            if self.token_rewards_as_adv:
                token_advantages = rolled_token_rewards * loss_mask
            else:
                adjacent_turn_tokens = token_loss_mask[:, :-1] & token_loss_mask[:, 1:]
                if turn_ids is not None:
                    raw_turn_ids = data["turn_ids"]
                    adjacent_turn_tokens &= raw_turn_ids[:, :-1] == raw_turn_ids[:, 1:]
                adjacent_rewards_match = (
                    aligned_token_rewards[:, :-1] == aligned_token_rewards[:, 1:]
                ) | ~adjacent_turn_tokens.to(device=aligned_token_rewards.device)
                torch._assert_async(
                    torch.all(adjacent_rewards_match),
                    "token_rewards_as_adv=False requires uniform token rewards "
                    "within each turn; use token_rewards_as_adv=True for dense "
                    "or sparse per-token rewards.",
                )
                next_mask = torch.zeros_like(loss_mask)
                next_mask[:, :-1] = loss_mask[:, 1:]
                if turn_ids is not None:
                    next_mask[:, :-1] *= turn_ids[:, :-1] == turn_ids[:, 1:]
                is_turn_end = loss_mask * (1 - next_mask)
                process_rewards = rolled_token_rewards * is_turn_end

        # Compute GAE.
        if "values" not in data:
            values = torch.zeros_like(rewards)
        else:
            values = data["values"]
        bootstrap_values = values[batch_indices, seqlens - 1]
        temporal_inputs = TemporalInputs(
            rewards=gae_outcome_rewards,
            kl_rewards=gae_kl_rewards,
            process_rewards=process_rewards,
            values=values,
            loss_mask=loss_mask,
            turn_ids=turn_ids,
            seq_no_eos_mask=seq_truncated_mask,
            bootstrap_values=bootstrap_values,
            discount=self.discount,
            gae_lambda=self.gae_lambda_fn
            if self._gae_lambda_is_custom
            else self.gae_lambda,
            gae_lambda_kwargs=self.gae_lambda_kwargs,
        )
        estimated = self.estimator(temporal_inputs)
        if not isinstance(estimated, TemporalResult):
            raise TypeError("estimator must return TemporalResult")
        for name in ("advantages", "returns", "total_rewards"):
            _validate_tensor(getattr(estimated, name), loss_mask, f"estimator {name}")
        advantages, returns = estimated.advantages, estimated.returns
        data["returns"] = returns

        # Optionally perform advantage normalization.
        if self.adv_norm is not None:
            # Use the same actual trajectory group sizes as reward normalization;
            # ignored when adv_norm is batch-level.
            advantages = self.adv_norm(
                advantages,
                loss_mask,
                group_sizes=group_sizes,
                group_member_counts=meta.logical_group_sizes
                if meta is not None
                else None,
            )

        advantages = process.func(
            ProcessAdvantageInputs(
                advantages=advantages,
                token_local_rewards=token_advantages,
                loss_mask=loss_mask,
                token_rewards_as_adv=self.token_rewards_as_adv,
                mask_no_eos_with_zero=self.mask_no_eos_with_zero,
            ),
            **process_kwargs,
        )
        _validate_tensor(advantages, loss_mask, "process advantages")

        return AdvantageResult(
            advantages=advantages,
            returns=returns,
            loss_mask=loss_mask,
            logprobs=old_logp,
            rewards=outcome.rewards,
            total_rewards=estimated.total_rewards,
            kl_rewards=kl_rewards,
            extras={
                key: data[key] for key in DERIVED_FIELDS if data.get(key) is not None
            },
        )
