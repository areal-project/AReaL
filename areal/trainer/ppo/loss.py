# SPDX-License-Identifier: Apache-2.0

"""Named token objectives using the existing optimizer-step reduction contract."""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from inspect import Parameter, signature
from types import MappingProxyType
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from areal.api.cli_args import LossTermConfig, PPOActorConfig, RejectionSamplingConfig
from areal.api.loss import named_loss_normalizer
from areal.trainer.ppo.loss_functions import LOSS_FUNCTIONS, mopd_loss, policy_loss
from areal.trainer.ppo.loss_reduction import (
    PG_TOKEN_WEIGHTS,
    PG_TURN_COUNTS,
    PreparedLossBatch,
    PreparedLossStep,
    prepare_policy_gradient_batch,
)
from areal.utils import stats_tracker
from areal.utils.constants import ProxLogpMethod
from areal.utils.dynamic_import import import_from_string


def _bind_loss_function(
    function: Callable[..., torch.Tensor],
    kwargs: dict[str, Any],
    config: PPOActorConfig,
) -> partial[torch.Tensor]:
    """Bind explicit built-in defaults, then validate the single plugin ABI."""
    runtime_names = {"outputs", "batch", "mask"}
    if runtime_names & kwargs.keys():
        raise ValueError("Loss kwargs cannot override outputs, batch, or mask")
    defaults = {}
    if function in {policy_loss}:
        defaults = {
            name: getattr(config, name)
            for name in (
                "eps_clip",
                "eps_clip_higher",
                "c_clip",
                "rejection_sampling",
                "importance_sampling_level",
                "prox_logp_method",
                "use_sapo_loss",
                "sapo_tau_pos",
                "sapo_tau_neg",
                "use_cispo_loss",
                "use_decoupled_loss",
            )
        }
    elif function is mopd_loss:
        defaults = {
            # Legacy SAPO does not narrow its teacher support via rejection.
            # Explicit per-term kwargs may still opt into independent filtering.
            "rejection_sampling": None
            if config.use_sapo_loss
            else config.rejection_sampling,
            "prox_logp_method": config.prox_logp_method,
        }
    options = defaults | kwargs
    if function in {policy_loss, mopd_loss}:
        # These options also control actor-side preprocessing/forward passes.
        # A per-term override cannot silently change the meaning of batch data.
        for name in ("prox_logp_method", "use_decoupled_loss"):
            if name in options and options[name] != getattr(config, name):
                raise ValueError(
                    f"Loss {name} must match actor.{name}; it controls batch preparation"
                )
        if (
            ProxLogpMethod(options["prox_logp_method"])
            == ProxLogpMethod.REUSE_TRAIN_LOGP
            and config.ppo_n_minibatches != 1
        ):
            raise ValueError("reuse_train_logp requires actor.ppo_n_minibatches=1")
        rejection = options.get("rejection_sampling")
        if isinstance(rejection, Mapping):
            fields = dict(rejection)
            # RPC DictConfig can contain the constructor's normalized clamp
            # default (lower=0). Rehydrate via its equivalent input lower=None.
            if fields.get("action") == "clamp" and fields.get("lower") == 0:
                fields["lower"] = None
            options["rejection_sampling"] = RejectionSamplingConfig(**fields)
        elif rejection is not None and not isinstance(
            rejection, RejectionSamplingConfig
        ):
            raise ValueError("Loss rejection_sampling must be a config mapping or None")
    if function in {policy_loss}:
        if "advantages" in kwargs:
            raise ValueError(
                "Loss advantages are runtime targets; configure advantages_key instead"
            )
        _validate_policy_options(options)
    validation_function = function
    try:
        parameters = signature(function).parameters
        if any(
            name not in parameters or parameters[name].kind != Parameter.KEYWORD_ONLY
            for name in runtime_names
        ):
            raise TypeError(
                "outputs, batch, mask must be explicit keyword-only parameters"
            )
        signature(validation_function).bind(
            outputs=None, batch=None, mask=None, **options
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid token loss {function!r}: expected loss_fn(*, outputs, batch, mask, "
            f"...algorithm_kwargs). {exc}"
        ) from exc
    return partial(function, **options)


def _validate_policy_options(options: dict[str, Any]) -> None:
    """Validate the effective per-term surrogate, not only actor defaults."""
    for name in ("use_sapo_loss", "use_cispo_loss", "use_decoupled_loss"):
        if not isinstance(options[name], bool):
            raise ValueError(f"Loss {name} must be boolean")
    if options["importance_sampling_level"] not in {"token", "sequence"}:
        raise ValueError("Loss importance_sampling_level must be token or sequence")
    if options["use_sapo_loss"] and options["use_cispo_loss"]:
        raise ValueError("CISPO and SAPO are mutually exclusive surrogates")
    for name in ("eps_clip", "eps_clip_higher", "c_clip"):
        value = options[name]
        if value is None and name != "eps_clip":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Loss {name} must be finite and nonnegative")
    if options["c_clip"] is not None and options["c_clip"] <= 1:
        raise ValueError("Loss c_clip must be greater than one")
    if "advantages_key" in options and (
        not isinstance(options["advantages_key"], str) or not options["advantages_key"]
    ):
        raise ValueError("Loss advantages_key must be a nonempty string")
    if options["use_sapo_loss"]:
        if options["use_decoupled_loss"]:
            raise ValueError("SAPO is not compatible with use_decoupled_loss=True")
        for name in ("sapo_tau_pos", "sapo_tau_neg"):
            value = options[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"SAPO {name} must be finite and positive")
    if options["use_cispo_loss"]:
        upper = options["eps_clip_higher"]
        if (
            isinstance(upper, bool)
            or not isinstance(upper, (int, float))
            or not math.isfinite(upper)
            or upper <= 0
        ):
            raise ValueError("CISPO requires a finite positive eps_clip_higher")
        if options["importance_sampling_level"] != "token":
            raise ValueError("CISPO only supports importance_sampling_level='token'")


@dataclass(frozen=True)
class ResolvedLoss:
    name: str
    function: Callable[..., torch.Tensor]
    coefficient: float
    reduction: str
    mask_key: str

    def key(self, suffix: str) -> str:
        return f"_actor_loss_{self.name}_{suffix}"

    def view(self, data: dict[str, Any]) -> dict[str, Any]:
        view = dict(data)
        view["loss_mask"] = data[self.key("support")]
        for key in (PG_TOKEN_WEIGHTS, PG_TURN_COUNTS):
            if self.key(key) in data:
                view[key] = data[self.key(key)]
        return view


@dataclass(frozen=True)
class ActorLossStep:
    terms: tuple[ResolvedLoss, ...]
    reductions: tuple[PreparedLossStep, ...]

    @property
    def loss_weight(self) -> Callable:
        return named_loss_normalizer(names=[term.name for term in self.terms])(
            partial(self.normalizers)
        )

    def normalizers(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            term.name: step.loss_weight(term.view(data))
            for term, step in zip(self.terms, self.reductions, strict=True)
        }

    def __call__(
        self,
        logprobs: torch.Tensor,
        entropy: torch.Tensor,
        input_data: dict[str, Any],
        *,
        current_version: int | None = None,
        **vocab_stats: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        outputs = MappingProxyType(
            {
                "logprobs": logprobs,
                "entropy": entropy,
                **{k: v for k, v in vocab_stats.items() if v is not None},
            }
        )
        batch = MappingProxyType(
            {
                **{
                    k: v
                    for k, v in input_data.items()
                    if not k.startswith("_actor_loss_")
                },
                "current_version": current_version,
            }
        )
        result = {}
        for term, step in zip(self.terms, self.reductions, strict=True):
            view = term.view(input_data)
            mask = view["loss_mask"]
            with stats_tracker.scope(term.name):
                values = term.function(outputs=outputs, batch=batch, mask=mask)
            if (
                not isinstance(values, torch.Tensor)
                or values.shape != mask.shape
                or values.device != mask.device
            ):
                raise ValueError(
                    "Token losses must match their support shape and device"
                )
            if (
                torch.is_grad_enabled()
                and logprobs.requires_grad
                and not values.requires_grad
            ):
                raise ValueError(
                    "Token losses must retain a differentiable connection to model outputs"
                )
            if not values.is_floating_point():
                raise ValueError("Token losses must be floating point")
            values = torch.where(mask, values, 0.0)
            torch._assert_async(
                torch.isfinite(values).all(),
                "Token losses must be finite on their support",
            )
            reduction = step.bind(view)
            mass = reduction.normalizer()
            mean = reduction.aggregate(values, mask)
            result[term.name] = mean * mass * term.coefficient
            with stats_tracker.scope(term.name):
                stats_tracker.weighted_mean(
                    "loss/avg", mean.detach() * term.coefficient, mass
                )
        return result


@dataclass(frozen=True)
class PreparedActorLoss:
    terms: tuple[ResolvedLoss, ...]
    batches: tuple[PreparedLossBatch, ...]

    def for_steps(
        self, microbatches: Sequence[dict[str, Any]], *, dp_group=None, device=None
    ) -> list[ActorLossStep]:
        steps = [
            batch.for_steps(microbatches, dp_group=dp_group, device=device)
            for batch in self.batches
        ]
        return [
            ActorLossStep(self.terms, tuple(column[i] for column in steps))
            for i in range(len(microbatches))
        ]


class ActorLoss:
    """Resolve plugins once; prepare each term before response scheduling."""

    def __init__(self, config: PPOActorConfig):
        self.config = config
        if config.m2_threshold is not None:
            raise ValueError(
                "Named losses do not support forward-dependent M2PO selection"
            )
        terms = []
        for name, raw in sorted(config.losses.items()):
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("loss term names must be Python identifiers")
            if isinstance(raw, DictConfig):
                raw = OmegaConf.to_container(raw, resolve=True)
            spec = (
                raw if isinstance(raw, LossTermConfig) else LossTermConfig(**dict(raw))
            )
            spec.__post_init__()
            if not spec.enabled or spec.coefficient == 0:
                continue
            function = LOSS_FUNCTIONS.get(spec.implementation)
            if function is None:
                if "." not in spec.implementation:
                    raise ValueError(
                        f"Unknown loss implementation {spec.implementation!r}"
                    )
                function = import_from_string(spec.implementation)
            if not callable(function):
                raise ValueError("Loss implementation must be callable")
            mode = spec.reduction or config.loss_aggregation
            if config.enable_tree_training and mode != "token_mean":
                raise ValueError("Tree training only supports token_mean losses")
            terms.append(
                ResolvedLoss(
                    name,
                    _bind_loss_function(function, spec.kwargs, config),
                    spec.coefficient,
                    mode,
                    spec.mask_key,
                )
            )
        if not terms:
            raise ValueError("At least one active loss term is required")
        self.terms = tuple(terms)

    def prepare(self, data: dict[str, Any], group_sizes=None) -> PreparedActorLoss:
        if data.get("teacher_logp") is not None:
            raise ValueError(
                "Explicit named losses do not compose legacy teacher_logp; use a named MOPD term"
            )
        prepared = []
        for term in self.terms:
            base = data["loss_mask"]
            mask = data.get(term.mask_key)
            if (
                not isinstance(mask, torch.Tensor)
                or mask.shape != base.shape
                or mask.device != base.device
            ):
                raise ValueError(
                    f"Loss mask {term.mask_key!r} must match loss_mask shape/device"
                )
            torch._assert_async(
                ((mask == 0) | (mask == 1)).all(), "Loss support masks must be binary"
            )
            support = base.bool() & mask.bool()
            data[term.key("support")] = support
            view = {**data, "loss_mask": support}
            prepared.append(
                prepare_policy_gradient_batch(
                    view,
                    mode=term.reduction,
                    divisor=self.config.loss_aggregation_divisor
                    if term.reduction == "constant"
                    else None,
                    group_sizes=group_sizes,
                    allow_empty_groups=True,
                )
            )
            for key in (PG_TOKEN_WEIGHTS, PG_TURN_COUNTS):
                if key in view:
                    data[term.key(key)] = view[key]
        return PreparedActorLoss(self.terms, tuple(prepared))
