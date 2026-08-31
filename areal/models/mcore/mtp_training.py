# SPDX-License-Identifier: Apache-2.0

"""Megatron-Core MTP supervision for AReaL actor training.

AReaL computes its policy objective from logits outside ``GPTModel``. Passing
``labels`` directly to Megatron would change that return value to the language
model loss. This module keeps AReaL's logits contract intact while exposing the
same labels and loss mask only to Megatron's ``process_mtp_loss`` helper.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu


@dataclass(frozen=True)
class MTPTrainingSupervision:
    """Next-token labels, actor mask, and main-loss-relative MTP scale."""

    labels: torch.Tensor
    loss_mask: torch.Tensor
    loss_multiplier: float | torch.Tensor = 1.0
    context_parallel: bool = False
    packed: bool = False


@dataclass(frozen=True)
class MTPCPRuntimeCapabilities:
    """Runtime features required by Megatron's packed MTP CP path."""

    missing: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return not self.missing


@dataclass(frozen=True)
class _MTPCPLossContext:
    """Collective used to normalize one CP-local MTP loss invocation."""

    group: dist.ProcessGroup
    world_size: int


class _MTPScaledConfig:
    """Delegate to a TransformerConfig while overriding only MTP loss scale."""

    def __init__(
        self,
        config: Any,
        loss_multiplier: float | torch.Tensor,
    ) -> None:
        self._config = config
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor * loss_multiplier

    def __getattr__(self, name: str) -> Any:
        return getattr(self._config, name)


_ACTIVE_MTP_SUPERVISION: ContextVar[MTPTrainingSupervision | None] = ContextVar(
    "areal_mcore_mtp_supervision",
    default=None,
)
_ACTIVE_MTP_CP_LOSS_CONTEXT: ContextVar[_MTPCPLossContext | None] = ContextVar(
    "areal_mcore_mtp_cp_loss_context",
    default=None,
)
_ACTIVE_MTP_DEPTH: ContextVar[int | None] = ContextVar(
    "areal_mcore_mtp_depth",
    default=None,
)
_MTP_BACKBONE_ONLY_FORWARD: ContextVar[bool] = ContextVar(
    "areal_mcore_mtp_backbone_only_forward",
    default=False,
)
_MTP_PROCESS_WRAPPER_MARKER = "_areal_mcore_mtp_process_wrapper"
_MTP_POSTPROCESS_WRAPPER_MARKER = "_areal_mcore_mtp_postprocess_wrapper"
_MTP_ROLL_WRAPPER_MARKER = "_areal_mcore_mtp_roll_wrapper"
_MTP_DETACH_WRAPPER_MARKER = "_areal_mcore_mtp_detach_wrapper"
_MTP_BLOCK_FORWARD_WRAPPER_MARKER = "_areal_mcore_mtp_block_forward_wrapper"
_MTP_CHECKPOINT_WRAPPER_MARKER = "_areal_mcore_mtp_checkpoint_wrapper"


@dataclass(frozen=True)
class _CheckpointTensorSlot:
    index: int


def _callable_parameters(value: Any) -> set[str]:
    if not callable(value):
        return set()
    try:
        return set(inspect.signature(value).parameters)
    except (TypeError, ValueError):
        return set()


def probe_mtp_cp_runtime(
    *,
    packed: bool,
    gpt_model_module: Any | None = None,
    mtp_module: Any | None = None,
) -> MTPCPRuntimeCapabilities:
    """Inspect the active Megatron APIs instead of relying on a version string."""

    if gpt_model_module is None:
        from megatron.core.models.gpt import gpt_model as gpt_model_module
    if mtp_module is None:
        from megatron.core.transformer import multi_token_prediction as mtp_module

    missing: list[str] = []
    roll_tensor = getattr(mtp_module, "roll_tensor", None)
    roll_parameters = _callable_parameters(roll_tensor)
    for parameter in ("cp_group", "packed_seq_params"):
        if parameter not in roll_parameters:
            missing.append(f"roll_tensor.{parameter}")
    if packed and not callable(getattr(mtp_module, "_roll_tensor_packed_seq", None)):
        missing.append("multi_token_prediction._roll_tensor_packed_seq")

    process_mtp_loss = getattr(gpt_model_module, "process_mtp_loss", None)
    process_parameters = _callable_parameters(process_mtp_loss)
    required_process_parameters = {
        "labels",
        "loss_mask",
        "output_layer",
        "output_weight",
        "compute_language_model_loss",
        "config",
        "cp_group",
        "packed_seq_params",
    }
    for parameter in sorted(required_process_parameters - process_parameters):
        missing.append(f"process_mtp_loss.{parameter}")

    unwrapped_process = (
        inspect.unwrap(process_mtp_loss) if callable(process_mtp_loss) else None
    )
    process_globals = getattr(unwrapped_process, "__globals__", {})
    if not callable(process_globals.get("roll_tensor")):
        missing.append("process_mtp_loss global roll_tensor")

    return MTPCPRuntimeCapabilities(missing=tuple(missing))


def require_mtp_cp_runtime(*, packed: bool) -> None:
    """Fail before training when Megatron cannot execute MTP with CP safely."""

    capabilities = probe_mtp_cp_runtime(packed=packed)
    if capabilities.supported:
        return
    raise RuntimeError(
        "MTP context parallelism requires Megatron-Core CP-aware MTP APIs; "
        "missing: " + ", ".join(capabilities.missing) + "."
    )


def build_mtp_supervision(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    loss_multiplier: float | torch.Tensor = 1.0,
) -> MTPTrainingSupervision:
    """Build next-token MTP labels without crossing sequence boundaries.

    ``loss_mask`` must already align with AReaL's normal next-token actor
    labels. Megatron rolls the returned labels and mask once more for each MTP
    prediction depth.
    """

    if input_ids.shape != loss_mask.shape:
        raise ValueError(
            "MTP input_ids and loss_mask must have the same shape, "
            f"got {tuple(input_ids.shape)} and {tuple(loss_mask.shape)}."
        )
    if input_ids.ndim not in (1, 2):
        raise ValueError(
            "MTP supervision supports packed [T] or padded [B, S] inputs, "
            f"got shape {tuple(input_ids.shape)}."
        )

    labels = torch.roll(input_ids, shifts=-1, dims=-1).to(dtype=torch.long).clone()
    aligned_loss_mask = loss_mask.float().clone()

    if cu_seqlens is not None:
        if input_ids.ndim != 1:
            raise ValueError(
                "Packed MTP input_ids must be one-dimensional before layout "
                f"conversion, got shape {tuple(input_ids.shape)}."
            )
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                "MTP cu_seqlens must be a one-dimensional boundary tensor."
            )
        sequence_ends = cu_seqlens[1:].to(dtype=torch.long) - 1
        labels[sequence_ends] = 0
        aligned_loss_mask[sequence_ends] = 0
    else:
        if input_ids.ndim != 2:
            raise ValueError("Non-packed MTP input_ids must use [B, S] layout.")
        labels[:, -1] = 0
        aligned_loss_mask[:, -1] = 0

    return MTPTrainingSupervision(
        labels=labels.contiguous(),
        loss_mask=aligned_loss_mask.contiguous(),
        loss_multiplier=loss_multiplier,
    )


def compute_mtp_loss_multiplier(
    local_weight: float | torch.Tensor,
    total_loss_weight: float | torch.Tensor,
    loss_multiplier: float | torch.Tensor,
    context_parallel_world_size: int,
) -> float | torch.Tensor:
    """Match AReaL's globally weighted main loss on the CP-local MTP path.

    ``total_loss_weight`` is reduced over DP x CP, so the same microbatch
    weight appears once per CP rank. The main policy loss cancels that factor
    through its full-sequence all-gather backward, while MTP consumes CP-local
    tokens and needs an explicit compensation. This factor is independent of
    the token-loss scaling that compensates the later CP gradient average.
    """

    if context_parallel_world_size < 1:
        raise ValueError("context_parallel_world_size must be positive.")
    return (
        local_weight / total_loss_weight * loss_multiplier * context_parallel_world_size
    )


@contextmanager
def mtp_supervision_context(
    supervision: MTPTrainingSupervision,
) -> Iterator[None]:
    """Expose one microbatch's supervision only during its model forward."""

    token = _ACTIVE_MTP_SUPERVISION.set(supervision)
    try:
        yield
    finally:
        _ACTIVE_MTP_SUPERVISION.reset(token)


@contextmanager
def mtp_backbone_only_context() -> Iterator[None]:
    """Skip the MTP block during logits-only actor forwards."""

    token = _MTP_BACKBONE_ONLY_FORWARD.set(True)
    try:
        yield
    finally:
        _MTP_BACKBONE_ONLY_FORWARD.reset(token)


def _call_output_layer_with_detached_weight(
    output_layer: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    bias = getattr(output_layer, "bias", None)
    if bias is not None and getattr(bias, "requires_grad", False):
        raise RuntimeError(
            "Isolated MTP training requires a bias-free or frozen LM head."
        )
    weight = kwargs.get("weight")
    if weight is None:
        weight = getattr(output_layer, "weight", None)
    if weight is None:
        raise RuntimeError(
            "Isolated MTP training requires an accessible LM-head weight."
        )
    kwargs["weight"] = weight.detach()
    return output_layer(*args, **kwargs)


def _wrap_roll_tensor_for_cp_loss(
    roll_tensor: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Use one CP-global denominator for the rolled floating-point loss mask."""

    if bool(getattr(roll_tensor, _MTP_ROLL_WRAPPER_MARKER, False)):
        return roll_tensor

    @functools.wraps(roll_tensor)
    def wrapped(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        rolled_tensor, token_count = roll_tensor(*args, **kwargs)
        loss_context = _ACTIVE_MTP_CP_LOSS_CONTEXT.get()
        if loss_context is None:
            return rolled_tensor, token_count

        tensor = kwargs.get("tensor", args[0] if args else None)
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            return rolled_tensor, token_count
        if not torch.is_tensor(token_count):
            token_count = rolled_tensor.sum(dtype=torch.float32)

        global_count = token_count.detach().to(dtype=torch.float32).clone()
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=loss_context.group)
        return rolled_tensor, global_count.clamp_min(1.0)

    setattr(wrapped, _MTP_ROLL_WRAPPER_MARKER, True)
    return wrapped


def _install_cp_loss_roll_hook(process_mtp_loss: Callable[..., Any]) -> None:
    """Patch the helper's module-level roll function with a context-local wrapper."""

    process_globals = getattr(inspect.unwrap(process_mtp_loss), "__globals__", None)
    if not isinstance(process_globals, dict):
        raise RuntimeError(
            "MTP CP loss normalization requires a Python process_mtp_loss helper."
        )
    roll_tensor = process_globals.get("roll_tensor")
    if not callable(roll_tensor):
        raise RuntimeError(
            "MTP CP loss normalization could not resolve process_mtp_loss.roll_tensor."
        )
    process_globals["roll_tensor"] = _wrap_roll_tensor_for_cp_loss(roll_tensor)


def _scale_token_loss_for_cp_average(
    compute_language_model_loss: Callable[..., torch.Tensor],
    cp_world_size: int,
) -> Callable[..., torch.Tensor]:
    """Cancel the CP parameter-gradient reduction after global normalization."""

    @functools.wraps(compute_language_model_loss)
    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        return compute_language_model_loss(*args, **kwargs) * cp_world_size

    return wrapped


def _wrap_process_mtp_loss(
    process_mtp_loss: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Inject AReaL actor supervision without changing Megatron's signature."""

    signature = inspect.signature(process_mtp_loss)
    required_parameters = {
        "labels",
        "loss_mask",
        "output_layer",
        "output_weight",
        "config",
    }
    missing = required_parameters.difference(signature.parameters)
    if missing:
        raise RuntimeError(
            "Unsupported Megatron process_mtp_loss signature; missing "
            f"parameters: {sorted(missing)}."
        )

    @functools.wraps(process_mtp_loss)
    def wrapped(*args: Any, **kwargs: Any) -> torch.Tensor:
        if _MTP_BACKBONE_ONLY_FORWARD.get():
            bound = signature.bind_partial(*args, **kwargs)
            hidden_states = bound.arguments.get("hidden_states")
            if hidden_states is None:
                raise RuntimeError("Backbone-only MTP forward requires hidden_states.")
            return hidden_states

        supervision = _ACTIVE_MTP_SUPERVISION.get()
        if supervision is None:
            return process_mtp_loss(*args, **kwargs)

        bound = signature.bind_partial(*args, **kwargs)
        if bound.arguments.get("labels") is not None:
            raise RuntimeError(
                "AReaL MTP supervision cannot be combined with labels passed "
                "directly to GPTModel."
            )
        config = bound.arguments.get("config")
        if config is None:
            raise RuntimeError("Megatron process_mtp_loss did not receive config.")

        cp_loss_context = None
        if supervision.context_parallel:
            required_cp_parameters = {
                "compute_language_model_loss",
                "cp_group",
                "packed_seq_params",
            }
            missing_cp_parameters = required_cp_parameters.difference(
                signature.parameters
            )
            if missing_cp_parameters:
                raise RuntimeError(
                    "MTP context parallelism requires a newer process_mtp_loss "
                    "signature; missing parameters: "
                    f"{sorted(missing_cp_parameters)}."
                )
            if bool(getattr(config, "calculate_per_token_loss", False)):
                raise NotImplementedError(
                    "MTP context parallelism does not support "
                    "calculate_per_token_loss=True because Megatron applies an "
                    "additional rank-local normalization in that mode."
                )
            cp_group = bound.arguments.get("cp_group")
            if cp_group is None:
                raise RuntimeError(
                    "MTP context parallelism requires process_mtp_loss.cp_group."
                )
            if supervision.packed and bound.arguments.get("packed_seq_params") is None:
                raise RuntimeError(
                    "Packed MTP context parallelism requires packed_seq_params."
                )
            compute_language_model_loss = bound.arguments.get(
                "compute_language_model_loss"
            )
            if not callable(compute_language_model_loss):
                raise RuntimeError(
                    "MTP CP loss normalization requires compute_language_model_loss."
                )
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "MTP context parallelism requires initialized torch.distributed."
                )
            cp_world_size = dist.get_world_size(group=cp_group)
            if cp_world_size <= 1:
                raise RuntimeError(
                    "MTP supervision is marked context-parallel but the CP group "
                    "has world size 1."
                )
            cp_loss_context = _MTPCPLossContext(
                group=cp_group,
                world_size=cp_world_size,
            )
            bound.arguments["compute_language_model_loss"] = (
                _scale_token_loss_for_cp_average(
                    compute_language_model_loss,
                    cp_world_size,
                )
            )
            _install_cp_loss_roll_hook(process_mtp_loss)

        bound.arguments["labels"] = supervision.labels
        bound.arguments["loss_mask"] = supervision.loss_mask
        bound.arguments["config"] = _MTPScaledConfig(
            config,
            supervision.loss_multiplier,
        )
        output_layer = bound.arguments.get("output_layer")
        if output_layer is None:
            raise RuntimeError(
                "Megatron process_mtp_loss did not receive output_layer."
            )
        bound.arguments["output_layer"] = functools.partial(
            _call_output_layer_with_detached_weight,
            output_layer,
        )
        output_weight = bound.arguments.get("output_weight")
        if output_weight is not None:
            bound.arguments["output_weight"] = output_weight.detach()
        cp_context_token = (
            _ACTIVE_MTP_CP_LOSS_CONTEXT.set(cp_loss_context)
            if cp_loss_context is not None
            else None
        )
        try:
            return process_mtp_loss(*bound.args, **bound.kwargs)
        finally:
            if cp_context_token is not None:
                _ACTIVE_MTP_CP_LOSS_CONTEXT.reset(cp_context_token)

    setattr(wrapped, _MTP_PROCESS_WRAPPER_MARKER, True)
    return wrapped


def _wrap_gpt_model_postprocess(
    postprocess: Callable[..., Any],
) -> Callable[..., Any]:
    """Prevent MTP computation during a logits-only actor forward."""

    signature = inspect.signature(postprocess)
    has_mtp_parameter = "mtp_in_postprocess" in signature.parameters
    accepts_keyword_arguments = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not has_mtp_parameter and not accepts_keyword_arguments:
        raise RuntimeError(
            "Unsupported Megatron GPTModel._postprocess signature; missing "
            "mtp_in_postprocess and **kwargs."
        )

    @functools.wraps(postprocess)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not _MTP_BACKBONE_ONLY_FORWARD.get():
            return postprocess(*args, **kwargs)
        if not has_mtp_parameter:
            kwargs["mtp_in_postprocess"] = False
            return postprocess(*args, **kwargs)
        bound = signature.bind_partial(*args, **kwargs)
        bound.arguments["mtp_in_postprocess"] = False
        return postprocess(*bound.args, **bound.kwargs)

    setattr(wrapped, _MTP_POSTPROCESS_WRAPPER_MARKER, True)
    return wrapped


def install_mtp_training_hook() -> bool:
    """Install the context-aware hook in the pinned Megatron GPT module."""

    from megatron.core.models.gpt import gpt_model

    installed = False
    current_process = gpt_model.process_mtp_loss
    if not bool(getattr(current_process, _MTP_PROCESS_WRAPPER_MARKER, False)):
        gpt_model.process_mtp_loss = _wrap_process_mtp_loss(current_process)
        installed = True

    current_postprocess = gpt_model.GPTModel._postprocess
    if not bool(getattr(current_postprocess, _MTP_POSTPROCESS_WRAPPER_MARKER, False)):
        gpt_model.GPTModel._postprocess = _wrap_gpt_model_postprocess(
            current_postprocess
        )
        installed = True
    return installed


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module"):
        model = model.module
    return getattr(model, "language_model", model)


def _resolve_mtp_layers(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    layers = getattr(getattr(_unwrap_model(model), "mtp", None), "layers", None)
    return tuple(layers) if layers else ()


def _resolve_mtp_block(model: torch.nn.Module) -> torch.nn.Module | None:
    return getattr(_unwrap_model(model), "mtp", None)


def _wrap_mtp_block_forward(forward: Callable[..., Any]) -> Callable[..., Any]:
    """Scope first-depth tracking without replacing the main hidden-state chunk."""

    @functools.wraps(forward)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if _ACTIVE_MTP_SUPERVISION.get() is None:
            return forward(*args, **kwargs)
        token = _ACTIVE_MTP_DEPTH.set(0)
        try:
            return forward(*args, **kwargs)
        finally:
            _ACTIVE_MTP_DEPTH.reset(token)

    setattr(wrapped, _MTP_BLOCK_FORWARD_WRAPPER_MARKER, True)
    return wrapped


def _install_concrete_postprocess_hook(model: torch.nn.Module) -> bool:
    """Wrap a model-specific ``_postprocess`` override when one exists.

    Megatron-Bridge models such as Qwen3.5 inherit from ``GPTModel`` but may
    install their own compatibility wrapper on the subclass. Patching the
    concrete class keeps logits-only forwards from running the MTP block even
    when that override captured the original base implementation.
    """

    model_class = type(_unwrap_model(model))
    current_postprocess = getattr(model_class, "_postprocess", None)
    if current_postprocess is None:
        raise RuntimeError(
            f"MTP model {model_class.__name__} has no _postprocess method."
        )
    if bool(getattr(current_postprocess, _MTP_POSTPROCESS_WRAPPER_MARKER, False)):
        return False
    model_class._postprocess = _wrap_gpt_model_postprocess(current_postprocess)
    return True


def _wrap_mtp_checkpointed_forward(
    checkpointed_forward: Callable[..., Any],
) -> Callable[..., Any]:
    """Keep PackedSeqParams out of MCore 0.17's tensor checkpoint inputs."""

    signature = inspect.signature(checkpointed_forward)
    parameters = tuple(signature.parameters.values())
    if (
        not parameters
        or parameters[0].name != "forward_func"
        or not any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        or not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
    ):
        raise RuntimeError(
            f"Unsupported Megatron MTP checkpoint signature: {signature}."
        )

    @functools.wraps(checkpointed_forward)
    def wrapped(
        forward_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        checkpoint_tensors: list[torch.Tensor] = []

        def capture(value: Any) -> Any:
            if not torch.is_tensor(value):
                return value
            slot = _CheckpointTensorSlot(len(checkpoint_tensors))
            checkpoint_tensors.append(value)
            return slot

        args_template = tuple(capture(value) for value in args)
        kwargs_template = {key: capture(value) for key, value in kwargs.items()}

        def restore(value: Any, values: tuple[torch.Tensor, ...]) -> Any:
            if isinstance(value, _CheckpointTensorSlot):
                return values[value.index]
            return value

        def tensor_only_forward(*values: torch.Tensor) -> Any:
            restored_args = tuple(restore(value, values) for value in args_template)
            restored_kwargs = {
                key: restore(value, values) for key, value in kwargs_template.items()
            }
            return forward_func(*restored_args, **restored_kwargs)

        return checkpointed_forward(tensor_only_forward, *checkpoint_tensors)

    setattr(wrapped, _MTP_CHECKPOINT_WRAPPER_MARKER, True)
    return wrapped


def configure_mtp_training(models: list[torch.nn.Module]) -> int:
    """Install hooks and isolate the auxiliary objective to MTP parameters."""

    if mpu.get_context_parallel_world_size() > 1:
        require_mtp_cp_runtime(packed=True)
        if any(
            bool(
                getattr(
                    getattr(_unwrap_model(model), "config", None),
                    "calculate_per_token_loss",
                    False,
                )
            )
            for model in models
        ):
            raise NotImplementedError(
                "MTP context parallelism does not support "
                "calculate_per_token_loss=True because Megatron applies an "
                "additional rank-local normalization in that mode."
            )
    install_mtp_training_hook()
    patched_layers = 0
    for model in models:
        _install_concrete_postprocess_hook(model)
        mtp_layers = _resolve_mtp_layers(model)
        if mtp_layers:
            mtp_block = _resolve_mtp_block(model)
            current_block_forward = getattr(mtp_block, "forward", None)
            if not callable(current_block_forward):
                raise RuntimeError(
                    "MTP model has layers but no callable block forward."
                )
            if not bool(
                getattr(
                    current_block_forward,
                    _MTP_BLOCK_FORWARD_WRAPPER_MARKER,
                    False,
                )
            ):
                mtp_block.forward = _wrap_mtp_block_forward(current_block_forward)

        for layer in mtp_layers:
            current_embeddings = layer._get_embeddings
            if not bool(getattr(current_embeddings, _MTP_DETACH_WRAPPER_MARKER, False)):
                embedding_signature = inspect.signature(current_embeddings)

                @functools.wraps(current_embeddings)
                def detached_get_embeddings(
                    *args: Any,
                    _original: Callable[..., Any] = current_embeddings,
                    _signature: inspect.Signature = embedding_signature,
                    **kwargs: Any,
                ) -> tuple[Any, ...]:
                    if _ACTIVE_MTP_SUPERVISION.get() is None:
                        return _original(*args, **kwargs)
                    bound = _signature.bind_partial(*args, **kwargs)
                    if bound.arguments.get("position_ids") is None:
                        input_ids = bound.arguments.get("input_ids")
                        if input_ids is None:
                            raise RuntimeError(
                                "MTP embedding hook did not receive input_ids."
                            )
                        bound.arguments["position_ids"] = torch.arange(
                            input_ids.shape[-1],
                            device=input_ids.device,
                            dtype=torch.long,
                        ).expand(input_ids.shape)
                    output = _original(*bound.args, **bound.kwargs)
                    if len(output) < 4:
                        raise RuntimeError(
                            "MTP _get_embeddings must return input IDs, position "
                            "IDs, decoder input, and hidden states."
                        )
                    *metadata, decoder_input, hidden_states = output
                    depth = _ACTIVE_MTP_DEPTH.get()
                    if depth is None:
                        raise RuntimeError(
                            "MTP depth tracking is inactive inside the MTP block."
                        )
                    _ACTIVE_MTP_DEPTH.set(depth + 1)
                    if depth == 0:
                        hidden_states = hidden_states.detach()
                        # Full activation checkpointing needs a differentiable
                        # input without reconnecting the backbone graph.
                        hidden_states.requires_grad_(True)
                    return (
                        *metadata,
                        decoder_input.detach(),
                        hidden_states,
                    )

                setattr(
                    detached_get_embeddings,
                    _MTP_DETACH_WRAPPER_MARKER,
                    True,
                )
                layer._get_embeddings = detached_get_embeddings
                patched_layers += 1

            current_checkpoint = layer._checkpointed_forward
            if bool(getattr(current_checkpoint, _MTP_CHECKPOINT_WRAPPER_MARKER, False)):
                continue
            layer._checkpointed_forward = _wrap_mtp_checkpointed_forward(
                current_checkpoint
            )

    return patched_layers


def collect_mtp_metrics(num_microbatches: int) -> dict[str, float]:
    """Drain Megatron's MTP tracker into AReaL train-batch statistics."""

    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive for MTP metrics.")

    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    total_loss_dict: dict[str, torch.Tensor] = {}
    MTPLossLoggingHelper.track_mtp_metrics(
        loss_scale=1.0 / num_microbatches,
        iteration=0,
        writer=None,
        wandb_writer=None,
        total_loss_dict=total_loss_dict,
    )
    return {
        f"mtp/{name.replace(' ', '_')}": float(value.detach().float().cpu().item())
        for name, value in total_loss_dict.items()
    }
