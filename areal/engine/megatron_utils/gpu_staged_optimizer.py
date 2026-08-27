# SPDX-License-Identifier: Apache-2.0

"""CPU-resident AdamW state with bounded GPU staging for Megatron-Core."""

from __future__ import annotations

import importlib.metadata
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.optim.adamw import adamw as functional_adamw

from areal.engine.megatron_utils.optimizer_chain import (
    iter_megatron_optimizer_leaves,
)
from areal.engine.megatron_utils.staged_optimizer_runtime import (
    CUDAStagingSlot,
    SlotStateMachine,
    StagedOptimizerRuntime,
)

_SUPPORTED_MEGATRON_CORE_VERSION = "0.17.0"
_FOREACH_MIN_ACTIVE_PARTS = 32

_PARAM_GROUP_OWNERSHIP_KEYS = (
    "wd_mult",
    "lr_mult",
    "is_expert_parallel",
    "is_decoupled_lr",
)


def _normalized_group_keys(group: Mapping[str, Any]) -> set[str]:
    keys = set(group)
    for key in _PARAM_GROUP_OWNERSHIP_KEYS:
        prefixed = f"pre_{key}"
        if prefixed in keys:
            keys.remove(prefixed)
            keys.add(key)
    return keys


def validate_managed_adamw_param_group(
    checkpoint_group: Mapping[str, Any],
    expected_group: Mapping[str, Any],
    *,
    location: str,
    ignore_params: bool,
) -> None:
    """Validate the one AdamW metadata contract used before and after DCP.

    MCore's outer state may spell ownership fields with a ``pre_`` prefix,
    while torch Optimizer state uses the canonical spelling.  Normalizing that
    representation here keeps the metadata-only preflight and the slab-backed
    inner load on exactly the same value-domain rules.
    """
    if not isinstance(checkpoint_group, Mapping):
        raise TypeError(f"{location} must be a mapping")
    if not isinstance(expected_group, Mapping):
        raise TypeError(f"{location} runtime metadata must be a mapping")

    checkpoint_keys = _normalized_group_keys(checkpoint_group)
    expected_keys = _normalized_group_keys(expected_group)
    if ignore_params:
        checkpoint_keys.discard("params")
        expected_keys.discard("params")
    if checkpoint_keys != expected_keys:
        missing = sorted(expected_keys - checkpoint_keys)
        extra = sorted(checkpoint_keys - expected_keys)
        raise ValueError(
            f"{location} metadata mismatch: missing={missing}, unexpected={extra}"
        )

    normalized: dict[str, Any] = {}
    for key, value in checkpoint_group.items():
        canonical = key.removeprefix("pre_") if key.startswith("pre_") else key
        if canonical in normalized:
            raise ValueError(f"{location} contains duplicate field {canonical!r}")
        normalized[canonical] = value
    expected = {
        (key.removeprefix("pre_") if key.startswith("pre_") else key): value
        for key, value in expected_group.items()
    }

    def finite_number(name: str) -> float:
        value = normalized[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{location} field {name} must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{location} field {name} must be finite")
        return result

    for name in ("lr", "initial_lr", "max_lr", "min_lr", "weight_decay"):
        if name in normalized and finite_number(name) < 0.0:
            raise ValueError(f"{location} field {name} must be non-negative")
    if "eps" in normalized and finite_number("eps") <= 0.0:
        raise ValueError(f"{location} field eps must be positive")
    for name in ("lr_mult", "wd_mult"):
        if name in normalized and finite_number(name) < 0.0:
            raise ValueError(f"{location} field {name} must be non-negative")

    if "betas" in normalized:
        betas = normalized["betas"]
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise TypeError(f"{location} field betas must be a pair")
        for beta in betas:
            if (
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not math.isfinite(float(beta))
                or not 0.0 <= float(beta) < 1.0
            ):
                raise ValueError(
                    f"{location} field betas must contain finite values in [0, 1)"
                )

    if "step" in normalized:
        step = normalized["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TypeError(f"{location} field step must be a non-negative int")

    for name in (
        "is_expert_parallel",
        "is_decoupled_lr",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    ):
        if name in normalized and (
            normalized[name] is not None and type(normalized[name]) is not bool
        ):
            raise TypeError(f"{location} field {name} must be bool or None")

    mutable = {
        "params",
        "lr",
        "initial_lr",
        "max_lr",
        "min_lr",
        "betas",
        "eps",
        "weight_decay",
        "step",
    }
    for name, value in normalized.items():
        if name in mutable:
            continue
        if name not in expected or value != expected[name]:
            raise ValueError(f"{location} field {name} does not match runtime metadata")


@dataclass(frozen=True)
class GPUStagedAdamWConfig:
    """Internal GPU staging configuration for CPU-resident AdamW."""

    buffer_count: int = 2
    bucket_size_mb: float = 128.0
    update_backend: Literal["auto", "single", "foreach", "fused"] = "auto"

    def __post_init__(self) -> None:
        if self.buffer_count < 1:
            raise ValueError("buffer_count must be at least 1")
        if self.bucket_size_mb <= 0:
            raise ValueError("bucket_size_mb must be positive")
        if self.update_backend not in {"auto", "single", "foreach", "fused"}:
            raise ValueError(f"unsupported AdamW update backend: {self.update_backend}")

    @property
    def bucket_numel(self) -> int:
        return max(1, int(self.bucket_size_mb * 1024 * 1024) // 4)


def _resolve_adamw_update_backend(
    config: GPUStagedAdamWConfig, active_part_count: int
) -> Literal["single", "foreach", "fused"]:
    if config.update_backend != "auto":
        return config.update_backend
    return "foreach" if active_part_count >= _FOREACH_MIN_ACTIVE_PARTS else "fused"


@dataclass(frozen=True)
class _ParamLayout:
    param: torch.nn.Parameter
    group_index: int
    offset: int
    numel: int


@dataclass(frozen=True)
class _UnitPart:
    param: torch.nn.Parameter
    param_offset: int
    unit_offset: int
    numel: int


@dataclass(frozen=True)
class _UpdateUnit:
    group_index: int
    slab_offset: int
    numel: int
    parts: tuple[_UnitPart, ...]


@dataclass
class AdamWCPUSlabs:
    """Three flat pinned FP32 slabs authoritative between optimizer steps."""

    master: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    @classmethod
    def allocate(cls, numel: int) -> AdamWCPUSlabs:
        kwargs = {
            "size": (numel,),
            "dtype": torch.float32,
            "device": "cpu",
            "pin_memory": True,
        }
        return cls(
            master=torch.empty(**kwargs),
            exp_avg=torch.zeros(**kwargs),
            exp_avg_sq=torch.zeros(**kwargs),
        )


class GPUStagedAdamW(torch.optim.AdamW):
    """AdamW whose FP32 master weights and moments live in pinned CPU slabs."""

    manages_cpu_residency = True
    manages_master_weight = True

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        staged_config: GPUStagedAdamWConfig | None = None,
        adam_w_mode: bool = True,
        master_weights: bool = True,
        use_decoupled_grad: bool = True,
        master_weight_dtype: torch.dtype = torch.float32,
        exp_avg_dtype: torch.dtype = torch.float32,
        exp_avg_sq_dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ) -> None:
        if not adam_w_mode:
            raise ValueError("GPU-staged optimizer only supports decoupled AdamW")
        if not master_weights or not use_decoupled_grad:
            raise ValueError(
                "GPU-staged AdamW requires precision-aware master weights and grads"
            )
        state_dtypes = (master_weight_dtype, exp_avg_dtype, exp_avg_sq_dtype)
        if any(dtype is not torch.float32 for dtype in state_dtypes):
            raise ValueError(
                "GPU-staged AdamW currently requires FP32 master and moment slabs"
            )
        unsupported_enabled = {
            name: kwargs.pop(name)
            for name in ("amsgrad", "maximize", "differentiable")
            if kwargs.get(name, False)
        }
        if unsupported_enabled:
            raise ValueError(
                f"unsupported AdamW options: {sorted(unsupported_enabled)}"
            )
        for ignored in (
            "capturable",
            "exp_avg_dtype",
            "exp_avg_sq_dtype",
            "store_param_remainders",
        ):
            kwargs.pop(ignored, None)
        if kwargs:
            raise TypeError(f"unexpected GPU-staged AdamW arguments: {sorted(kwargs)}")

        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            fused=False,
        )
        self.staged_config = staged_config or GPUStagedAdamWConfig()
        self.cpu_slabs: AdamWCPUSlabs | None = None
        self._layouts: tuple[_ParamLayout, ...] = ()
        self._units: tuple[_UpdateUnit, ...] = ()
        self._runtime = StagedOptimizerRuntime(
            self.staged_config.buffer_count,
            ("master", "exp_avg", "exp_avg_sq", "grad"),
        )
        self._bound = False

    @property
    def residency(self) -> str:
        return self._runtime.residency

    @property
    def _slots(self) -> list[CUDAStagingSlot]:
        return self._runtime.slots

    @property
    def _slot_machine(self) -> SlotStateMachine | None:
        return self._runtime.slot_machine

    @property
    def checkpoint_lifecycle(self) -> str:
        return "FAILED" if self._runtime.checkpoint_load_error is not None else "IDLE"

    @property
    def units(self) -> tuple[_UpdateUnit, ...]:
        return self._units

    @property
    def gpu_staging_state_numel(self) -> int:
        return sum(
            slot.master.numel() + slot.exp_avg.numel() + slot.exp_avg_sq.numel()
            for slot in self._slots
        )

    @property
    def cuda_state_numel(self) -> int:
        """CUDA tensor elements in authoritative optimizer state (slots excluded)."""
        return sum(
            value.numel()
            for state in self.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_cuda
        )

    def initialize_state(self, param: torch.Tensor) -> None:
        """Compatibility hook used by Megatron's precision-aware init callback."""
        if self._bound and param not in self.state:
            raise KeyError("parameter is not owned by this staged optimizer")

    def bind_owned_params(
        self, param_groups: list[dict[str, Any]], **metadata: Any
    ) -> None:
        """Bind final DP-local shards and initialize their CPU-authoritative state."""
        empty_device = metadata.pop("empty_device", None)
        # DistributedOptimizer supplies ownership metadata used by checkpoint
        # identity construction.  The AdamW layout itself is already encoded
        # in the final param groups, so those fields remain intentionally opaque.
        del metadata
        if self._bound:
            raise RuntimeError("GPU-staged AdamW is already bound")
        if len(param_groups) != len(self.param_groups):
            raise ValueError("bound parameter groups do not match optimizer groups")
        self.param_groups = param_groups

        layouts: list[_ParamLayout] = []
        total_numel = 0
        devices: set[torch.device] = set()
        for group_index, group in enumerate(self.param_groups):
            group.setdefault("step", 0)
            for param in group["params"]:
                if not param.is_cuda:
                    raise ValueError("GPU-staged AdamW parameters must be CUDA tensors")
                devices.add(param.device)
                layouts.append(
                    _ParamLayout(param, group_index, total_numel, param.numel())
                )
                total_numel += param.numel()
        if len(devices) > 1:
            raise ValueError(
                "all parameters owned by one staged optimizer must share a CUDA device"
            )
        if not devices and empty_device is None:
            raise ValueError(
                "empty staged AdamW ownership requires an explicit CUDA device"
            )

        self.cpu_slabs = AdamWCPUSlabs.allocate(total_numel)
        self._layouts = tuple(layouts)
        self._units = self._build_units(layouts, self.staged_config.bucket_numel)
        device = next(iter(devices), empty_device)
        assert device is not None
        if device.type != "cuda":
            raise ValueError("staged AdamW device must be CUDA")
        capacity = max((unit.numel for unit in self._units), default=None)
        self._runtime.bind(capacity=capacity, device=device)

        self.state.clear()
        for layout in self._layouts:
            state = self.state[layout.param]
            state["master_param"] = self.cpu_slabs.master.narrow(
                0, layout.offset, layout.numel
            ).view_as(layout.param)
            state["exp_avg"] = self.cpu_slabs.exp_avg.narrow(
                0, layout.offset, layout.numel
            ).view_as(layout.param)
            state["exp_avg_sq"] = self.cpu_slabs.exp_avg_sq.narrow(
                0, layout.offset, layout.numel
            ).view_as(layout.param)

        self._bound = True
        if self._units:
            self._runtime.schedule_units(
                self._units,
                self._schedule_master_initialization,
                wait_for_compute=False,
            )
        self.drain()

    @staticmethod
    def _build_units(
        layouts: list[_ParamLayout], bucket_numel: int
    ) -> tuple[_UpdateUnit, ...]:
        units: list[_UpdateUnit] = []
        by_group: dict[int, list[_ParamLayout]] = {}
        for layout in layouts:
            by_group.setdefault(layout.group_index, []).append(layout)
        for group_index, group_layouts in by_group.items():
            group_start = group_layouts[0].offset
            group_end = group_layouts[-1].offset + group_layouts[-1].numel
            unit_start = group_start
            while unit_start < group_end:
                unit_end = min(unit_start + bucket_numel, group_end)
                parts: list[_UnitPart] = []
                for layout in group_layouts:
                    overlap_start = max(unit_start, layout.offset)
                    overlap_end = min(unit_end, layout.offset + layout.numel)
                    if overlap_end > overlap_start:
                        parts.append(
                            _UnitPart(
                                param=layout.param,
                                param_offset=overlap_start - layout.offset,
                                unit_offset=overlap_start - unit_start,
                                numel=overlap_end - overlap_start,
                            )
                        )
                units.append(
                    _UpdateUnit(
                        group_index=group_index,
                        slab_offset=unit_start,
                        numel=unit_end - unit_start,
                        parts=tuple(parts),
                    )
                )
                unit_start = unit_end
        return tuple(units)

    def _schedule_master_initialization(
        self,
        unit: _UpdateUnit,
        slot_index: int,
        params_ready: torch.cuda.Event,
    ) -> None:
        assert self.cpu_slabs is not None
        slot = self._runtime.acquire_slot(slot_index)
        with torch.cuda.stream(slot.h2d_stream):
            slot.h2d_stream.wait_event(params_ready)
            for part in unit.parts:
                slot.master.narrow(0, part.unit_offset, part.numel).copy_(
                    part.param.detach()
                    .view(-1)
                    .narrow(0, part.param_offset, part.numel)
                )
            slot.h2d_done.record(slot.h2d_stream)
        with torch.cuda.stream(slot.d2h_stream):
            slot.d2h_stream.wait_event(slot.h2d_done)
            self.cpu_slabs.master.narrow(0, unit.slab_offset, unit.numel).copy_(
                slot.master.narrow(0, 0, unit.numel), non_blocking=True
            )
            slot.d2h_done.record(slot.d2h_stream)
        self._runtime.mark_d2h_pending(slot_index)

    def _schedule_update(
        self,
        unit: _UpdateUnit,
        slot_index: int,
        grads_ready: torch.cuda.Event,
    ) -> None:
        assert self.cpu_slabs is not None
        slot = self._runtime.acquire_slot(slot_index)
        count = unit.numel
        slab_slice = slice(unit.slab_offset, unit.slab_offset + count)

        with torch.cuda.stream(slot.h2d_stream):
            # The caller stream contains gradient finalize/reduce-scatter,
            # overflow, norm and clipping work that precedes inner step().
            slot.h2d_stream.wait_event(grads_ready)
            slot.master[:count].copy_(
                self.cpu_slabs.master[slab_slice], non_blocking=True
            )
            slot.exp_avg[:count].copy_(
                self.cpu_slabs.exp_avg[slab_slice], non_blocking=True
            )
            slot.exp_avg_sq[:count].copy_(
                self.cpu_slabs.exp_avg_sq[slab_slice], non_blocking=True
            )
            slot.h2d_done.record(slot.h2d_stream)

        group = self.param_groups[unit.group_index]
        beta1, beta2 = group["betas"]
        step = int(group["step"])
        lr = float(group["lr"])
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        with torch.cuda.stream(slot.compute_stream):
            slot.compute_stream.wait_event(grads_ready)
            slot.compute_stream.wait_event(slot.h2d_done)
            active_parts: list[_UnitPart] = []
            master_parts: list[torch.Tensor] = []
            grad_parts: list[torch.Tensor] = []
            exp_avg_parts: list[torch.Tensor] = []
            exp_avg_sq_parts: list[torch.Tensor] = []
            for part in unit.parts:
                grad = getattr(part.param, "decoupled_grad", None)
                if grad is None:
                    grad = part.param.grad
                if grad is None:
                    continue
                slot.grad.narrow(0, part.unit_offset, part.numel).copy_(
                    grad.detach().view(-1).narrow(0, part.param_offset, part.numel)
                )
                active_parts.append(part)
                master_parts.append(slot.master.narrow(0, part.unit_offset, part.numel))
                grad_parts.append(slot.grad.narrow(0, part.unit_offset, part.numel))
                exp_avg_parts.append(
                    slot.exp_avg.narrow(0, part.unit_offset, part.numel)
                )
                exp_avg_sq_parts.append(
                    slot.exp_avg_sq.narrow(0, part.unit_offset, part.numel)
                )

            if active_parts:
                backend = _resolve_adamw_update_backend(
                    self.staged_config, len(active_parts)
                )
                use_foreach = backend == "foreach"
                use_fused = backend == "fused"
                # Functional AdamW increments each supplied state step. Every
                # slice in this unit therefore receives a disposable copy of
                # the Megatron group step immediately before this update.
                step_device = master_parts[0].device if use_fused else None
                state_steps = [
                    torch.tensor(
                        float(step - 1), dtype=torch.float32, device=step_device
                    )
                    for _ in active_parts
                ]
                functional_adamw(
                    master_parts,
                    grad_parts,
                    exp_avg_parts,
                    exp_avg_sq_parts,
                    [],
                    state_steps,
                    foreach=use_foreach,
                    capturable=False,
                    differentiable=False,
                    fused=use_fused,
                    grad_scale=None,
                    found_inf=None,
                    has_complex=False,
                    amsgrad=False,
                    beta1=beta1,
                    beta2=beta2,
                    lr=lr,
                    weight_decay=weight_decay,
                    eps=eps,
                    maximize=False,
                )
            for part, master_part in zip(active_parts, master_parts, strict=True):
                part.param.detach().view(-1).narrow(
                    0, part.param_offset, part.numel
                ).copy_(master_part)
            slot.compute_done.record(slot.compute_stream)

        with torch.cuda.stream(slot.d2h_stream):
            slot.d2h_stream.wait_event(slot.compute_done)
            self.cpu_slabs.master[slab_slice].copy_(
                slot.master[:count], non_blocking=True
            )
            self.cpu_slabs.exp_avg[slab_slice].copy_(
                slot.exp_avg[:count], non_blocking=True
            )
            self.cpu_slabs.exp_avg_sq[slab_slice].copy_(
                slot.exp_avg_sq[:count], non_blocking=True
            )
            slot.d2h_done.record(slot.d2h_stream)
        self._runtime.mark_d2h_pending(slot_index)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if not self._bound:
            raise RuntimeError("bind_owned_params() must be called before step()")
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "GPU-staged AdamW is unavailable after a failed checkpoint load"
            ) from self._runtime.checkpoint_load_error
        loss = closure() if closure is not None else None
        if not self._units:
            self._runtime.residency = "CPU_RESIDENT"
            return loss
        for group in self.param_groups:
            if group["params"]:
                group["step"] = int(group.get("step", 0)) + 1
        # Megatron starts parameter all-gather immediately after inner step, so
        # the shared runtime queues compute visibility on the calling stream.
        self._runtime.schedule_units(
            self._units, self._schedule_update, wait_for_compute=True
        )
        return loss

    def drain(self) -> None:
        self._runtime.drain()

    def offload_to_cpu(self) -> None:
        self.drain()
        # The CPU slabs are authoritative between steps.  Keeping the bounded
        # CUDA staging slots alive while AWEX hands the device to rollout only
        # wastes colocation headroom (2 GiB with the default 2 x 128 MiB x 4
        # tensors).  Streams/events are slot-owned as well, so dropping the
        # slots releases the complete staging runtime.
        self._runtime.release_slots()

    def restore_from_cpu(self) -> None:
        if not self._bound or not self._units or self._slots:
            return
        capacity = max(unit.numel for unit in self._units)
        device = self._layouts[0].param.device
        self._runtime.restore_slots(capacity=capacity, device=device)

    def prepare_checkpoint_save(self) -> None:
        """Drain GPU staging before synchronous DCP reads the CPU slabs."""
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "cannot save after a failed checkpoint load; restart and recover"
            ) from self._runtime.checkpoint_load_error
        if not self._bound or self.cpu_slabs is None:
            raise RuntimeError("GPU-staged AdamW must be bound before checkpoint save")
        self.drain()
        if self.cuda_state_numel != 0:
            raise RuntimeError("staged optimizer checkpoint source contains CUDA state")
        self._validate_bound_state_views()

    def begin_checkpoint_load(self) -> None:
        """Prepare for an in-place synchronous load without rollback state."""
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "checkpoint load already failed; restart the process to recover"
            ) from self._runtime.checkpoint_load_error
        if not self._bound:
            raise RuntimeError("GPU-staged AdamW must be bound before checkpoint load")
        self.drain()

    def complete_checkpoint_load(self) -> None:
        """Validate the newly loaded CPU-authoritative optimizer state."""
        self.drain()
        self._validate_bound_state_views()

    def mark_checkpoint_load_failed(self, error: BaseException) -> None:
        """Fail-stop after an in-place load error; no current-process rollback."""
        self._runtime.mark_checkpoint_load_failed(error)

    def reset_from_model_params(self) -> None:
        """Rebuild masters and clear moments after a model-only checkpoint load."""
        if not self._bound or self.cpu_slabs is None:
            raise RuntimeError("GPU-staged AdamW must be bound before state reset")
        slots_were_released = not self._slots
        if slots_were_released:
            self.restore_from_cpu()
        try:
            self.drain()
            if self._units:
                self._runtime.schedule_units(
                    self._units,
                    self._schedule_master_initialization,
                    wait_for_compute=False,
                )
                self.drain()
            self.cpu_slabs.exp_avg.zero_()
            self.cpu_slabs.exp_avg_sq.zero_()
            for group in self.param_groups:
                group["step"] = 0
        finally:
            if slots_were_released:
                self.offload_to_cpu()

    def get_unscaled_state(self, param: torch.Tensor, key: str) -> torch.Tensor:
        self.prepare_checkpoint_save()
        return self.state[param][key]

    def set_scaled_state(
        self, param: torch.Tensor, key: str, value: torch.Tensor
    ) -> None:
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "cannot mutate optimizer after a failed checkpoint load"
            ) from self._runtime.checkpoint_load_error
        self.drain()
        if key not in ("master_param", "exp_avg", "exp_avg_sq"):
            raise KeyError(f"unsupported staged AdamW checkpoint state: {key}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"loaded {key} must be a tensor")
        if value.device.type != "cpu" or value.dtype is not torch.float32:
            raise TypeError(f"loaded {key} must be a CPU FP32 tensor")
        destination = self.state[param][key]
        if destination.shape != value.shape:
            raise ValueError(
                f"checkpoint state shape mismatch for {key}: "
                f"expected {tuple(destination.shape)}, got {tuple(value.shape)}"
            )
        destination.copy_(value)

    def state_dict(self) -> dict[str, Any]:
        if self._bound:
            self.prepare_checkpoint_save()
        return super().state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load without torch Optimizer casting CPU state to the parameter device."""
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "cannot reload optimizer after a failed checkpoint load"
            ) from self._runtime.checkpoint_load_error
        if not self._bound:
            self._load_unbound_metadata_state_dict(state_dict)
            return
        loaded_groups, id_to_param = self._validate_state_dict_schema(state_dict)
        self.drain()
        for current_group, loaded_group in zip(self.param_groups, loaded_groups):
            current_params = current_group["params"]
            current_group.clear()
            current_group.update(
                {key: value for key, value in loaded_group.items() if key != "params"}
            )
            current_group["params"] = current_params
        for loaded_id, loaded_state in state_dict["state"].items():
            param = id_to_param[loaded_id]
            for key in ("master_param", "exp_avg", "exp_avg_sq"):
                self.state[param][key].copy_(loaded_state[key])

    def _load_unbound_metadata_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "state",
            "param_groups",
        }:
            raise KeyError(
                "unbound optimizer state must contain state and param_groups"
            )
        if state_dict["state"]:
            raise RuntimeError("unbound GPU-staged AdamW cannot load tensor state")
        loaded_groups = state_dict["param_groups"]
        if not isinstance(loaded_groups, (list, tuple)) or len(loaded_groups) != len(
            self.param_groups
        ):
            raise ValueError("unbound optimizer parameter groups do not match")
        for group_index, (current_group, loaded_group) in enumerate(
            zip(self.param_groups, loaded_groups)
        ):
            if not isinstance(loaded_group, Mapping) or set(loaded_group) != set(
                current_group
            ):
                raise KeyError(
                    f"unbound optimizer param_group {group_index} fields do not match"
                )
            if len(loaded_group["params"]) != len(current_group["params"]):
                raise ValueError(
                    "unbound optimizer parameter group size does not match"
                )
            self._validate_param_group_metadata(
                current_group, loaded_group, group_index
            )
        for current_group, loaded_group in zip(self.param_groups, loaded_groups):
            params = current_group["params"]
            current_group.clear()
            current_group.update(
                {key: value for key, value in loaded_group.items() if key != "params"}
            )
            current_group["params"] = params

    def _validate_state_dict_schema(
        self, state_dict: Mapping[str, Any]
    ) -> tuple[list[Mapping[str, Any]], dict[int, torch.Tensor]]:
        if not isinstance(state_dict, Mapping):
            raise TypeError("optimizer checkpoint must be a mapping")
        expected_top = {"state", "param_groups"}
        if set(state_dict) != expected_top:
            raise KeyError("optimizer checkpoint top-level fields do not match")
        if self.cpu_slabs is None:
            raise RuntimeError("GPU-staged AdamW must be bound before state load")
        loaded_groups_value = state_dict["param_groups"]
        if not isinstance(loaded_groups_value, (list, tuple)):
            raise TypeError("optimizer param_groups must be a list or tuple")
        loaded_groups = list(loaded_groups_value)
        if len(loaded_groups) != len(self.param_groups):
            raise ValueError("loaded optimizer has a different number of groups")

        id_to_param: dict[int, torch.Tensor] = {}
        for group_index, (current_group, loaded_group) in enumerate(
            zip(self.param_groups, loaded_groups)
        ):
            if not isinstance(loaded_group, Mapping):
                raise TypeError(
                    f"optimizer param_group {group_index} must be a mapping"
                )
            if set(loaded_group) != set(current_group):
                raise KeyError(
                    f"optimizer param_group {group_index} fields do not match"
                )
            loaded_ids = loaded_group["params"]
            if not isinstance(loaded_ids, (list, tuple)) or len(loaded_ids) != len(
                current_group["params"]
            ):
                raise ValueError("loaded optimizer parameter group size does not match")
            for loaded_id, param in zip(loaded_ids, current_group["params"]):
                if not isinstance(loaded_id, int) or isinstance(loaded_id, bool):
                    raise TypeError("optimizer parameter identifiers must be integers")
                if loaded_id in id_to_param:
                    raise ValueError(
                        f"optimizer parameter identifier {loaded_id} is duplicated"
                    )
                id_to_param[loaded_id] = param
            self._validate_param_group_metadata(
                current_group, loaded_group, group_index
            )

        loaded_state = state_dict["state"]
        if not isinstance(loaded_state, Mapping) or set(loaded_state) != set(
            id_to_param
        ):
            raise ValueError("loaded optimizer state parameter set does not match")
        expected_keys = {"master_param", "exp_avg", "exp_avg_sq"}
        for loaded_id, values in loaded_state.items():
            if not isinstance(values, Mapping) or set(values) != expected_keys:
                raise KeyError(
                    f"optimizer state for parameter {loaded_id} fields do not match"
                )
            param = id_to_param[loaded_id]
            for key in expected_keys:
                value = values[key]
                destination = self.state[param][key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.dtype is not torch.float32
                    or value.shape != destination.shape
                ):
                    raise ValueError(f"loaded {key} has an incompatible tensor")
        self._validate_bound_state_views()
        return loaded_groups, id_to_param

    @staticmethod
    def _validate_param_group_metadata(
        current: Mapping[str, Any], loaded: Mapping[str, Any], group_index: int
    ) -> None:
        validate_managed_adamw_param_group(
            loaded,
            current,
            location=f"optimizer param_group {group_index}",
            ignore_params=False,
        )

    def _validate_bound_state_views(self) -> None:
        assert self.cpu_slabs is not None
        slabs = {
            "master_param": self.cpu_slabs.master,
            "exp_avg": self.cpu_slabs.exp_avg,
            "exp_avg_sq": self.cpu_slabs.exp_avg_sq,
        }
        for layout in self._layouts:
            state = self.state.get(layout.param)
            if not isinstance(state, Mapping) or set(state) != set(slabs):
                raise RuntimeError(
                    "managed optimizer state schema no longer matches slabs"
                )
            for key, slab in slabs.items():
                value = state[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.dtype is not torch.float32
                    or value.shape != layout.param.shape
                    or value.storage_offset() != slab.storage_offset() + layout.offset
                    or value.untyped_storage().data_ptr()
                    != slab.untyped_storage().data_ptr()
                ):
                    raise RuntimeError(
                        f"managed optimizer {key} view lost CPU FP32 slab ownership"
                    )


def _check_megatron_compatibility() -> None:
    version = importlib.metadata.version("megatron-core")
    if version != _SUPPORTED_MEGATRON_CORE_VERSION:
        raise RuntimeError(
            "GPU-staged AdamW compatibility layer supports megatron-core "
            f"{_SUPPORTED_MEGATRON_CORE_VERSION}, found {version}"
        )


def _iter_megatron_optimizers(optimizer: Any) -> Iterator[Any]:
    yield from iter_megatron_optimizer_leaves(optimizer)


def bind_gpu_staged_adamw(optimizer: Any) -> int:
    """Bind all managed inner optimizers after MCore has established DP shards."""
    bound = 0
    for megatron_optimizer in _iter_megatron_optimizers(optimizer):
        inner = getattr(megatron_optimizer, "optimizer", None)
        if not getattr(inner, "manages_cpu_residency", False):
            continue
        inner.bind_owned_params(
            [group["orig_group"] for group in megatron_optimizer.opt_group_ranges],
            gbuf_ranges=megatron_optimizer.gbuf_ranges,
            model_param_gbuf_map=megatron_optimizer.model_param_gbuf_map,
            buffers=megatron_optimizer.buffers,
        )
        bound += 1
    return bound


def _replace_metadata_optimizers_with_staged_adamw(
    optimizer: Any,
    mcore_config: Any,
    staged_config: GPUStagedAdamWConfig,
) -> int:
    """Replace only already-built DP optimizer instances, never MCore globals."""
    replaced = 0
    for megatron_optimizer in _iter_megatron_optimizers(optimizer):
        inner = getattr(megatron_optimizer, "optimizer", None)
        if inner is None or getattr(megatron_optimizer, "is_stub_optimizer", False):
            continue
        if getattr(inner, "manages_cpu_residency", False):
            raise RuntimeError("Megatron optimizer is already residency-managed")
        if len(inner.state) != 0:
            raise RuntimeError(
                "MCore Adam allocated tensor state before staged ownership binding"
            )
        staged = GPUStagedAdamW(
            inner.param_groups,
            lr=mcore_config.lr,
            betas=(mcore_config.adam_beta1, mcore_config.adam_beta2),
            eps=mcore_config.adam_eps,
            weight_decay=mcore_config.weight_decay,
            staged_config=staged_config,
            adam_w_mode=mcore_config.decoupled_weight_decay,
            master_weights=True,
            use_decoupled_grad=True,
            master_weight_dtype=mcore_config.main_params_dtype,
            exp_avg_dtype=mcore_config.exp_avg_dtype,
            exp_avg_sq_dtype=mcore_config.exp_avg_sq_dtype,
        )
        megatron_optimizer.optimizer = staged
        replaced += 1
    return replaced


def get_megatron_optimizer_with_gpu_staged_adamw(
    mcore_config: Any,
    model: list[Any],
    staged_config: GPUStagedAdamWConfig,
) -> Any:
    """Build through MCore, then bind CPU slabs to its final DP-local shards."""
    _check_megatron_compatibility()
    required = {
        "optimizer": "adam",
        "use_distributed_optimizer": True,
        "bf16": True,
        "optimizer_cpu_offload": False,
        "use_precision_aware_optimizer": True,
        "main_params_dtype": torch.float32,
        "exp_avg_dtype": torch.float32,
        "exp_avg_sq_dtype": torch.float32,
    }
    mismatches = {
        name: (getattr(mcore_config, name, None), expected)
        for name, expected in required.items()
        if getattr(mcore_config, name, None) != expected
    }
    if mismatches:
        raise ValueError(f"incompatible Megatron optimizer config: {mismatches}")
    if getattr(mcore_config, "optimizer_cuda_graph", False):
        raise ValueError("GPU-staged AdamW does not support optimizer CUDA graphs")
    if getattr(mcore_config, "fp8_recipe", None) is not None:
        raise ValueError("GPU-staged AdamW first stage supports BF16 without FP8")

    import megatron.core.optimizer as mcore_optimizer

    # MCore's precision-aware Adam constructor is metadata-only here.  Let it
    # establish process groups and DP-local parameter shards, then replace only
    # the resulting wrapper instances.  The module-global Adam class is never
    # read-modify-written, so staged and ordinary builders can run concurrently.
    optimizer = mcore_optimizer.get_megatron_optimizer(mcore_config, model)
    replaced = _replace_metadata_optimizers_with_staged_adamw(
        optimizer, mcore_config, staged_config
    )
    if replaced == 0 or bind_gpu_staged_adamw(optimizer) != replaced:
        raise RuntimeError(
            "Megatron builder did not produce a managed distributed optimizer"
        )
    return optimizer
