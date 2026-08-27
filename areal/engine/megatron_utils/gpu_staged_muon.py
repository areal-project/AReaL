# SPDX-License-Identifier: Apache-2.0

"""MCore 0.17 layer-wise Muon with CPU-authoritative optimizer state.

The official Muon builder remains responsible for parameter classification and
``LayerWiseDistributedOptimizer.shard_params`` ownership.  Only after that
ownership decision do we replace each rank-local leaf with a staged backend.
Muon matrices are indivisible units; scalar parameters retain the staged AdamW
backend.  Synchronous checkpointing permits DP/expert-DP ownership resharding
while keeping every model-parallel dimension fixed. Async checkpoint and
prefetch are deliberately outside this MVP.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from areal.engine.megatron_utils.gpu_staged_optimizer import (
    GPUStagedAdamW,
    GPUStagedAdamWConfig,
)
from areal.engine.megatron_utils.staged_optimizer_runtime import (
    CUDAStagingSlot,
    SlotStateMachine,
    StagedOptimizerRuntime,
)

_SUPPORTED_MEGATRON_CORE_VERSION = "0.17.0"
_SUPPORTED_EMERGING_OPTIMIZERS_VERSION = "0.3.0"
_MCORE017_LAYERWISE_ALLGATHER_SOURCE_SHA256 = (
    "8a8f3d3d914a38665deba909917cf30e92a69d60d08bd0af0a5d7eaed7e3c846"
)
_EMPTY_MUON_LEAF = object()
_MUON_CHECKPOINT_SCHEMA_VERSION = 2
_MUON_CHECKPOINT_PREFIX = "optimizer.gpu_staged_muon.v2"
_MUON_CHECKPOINT_STATE_KINDS = {
    "muon": ("master_param", "momentum_buffer"),
    "scalar_adamw": ("master_param", "exp_avg", "exp_avg_sq"),
}


def _require_muon_checkpoint_versions() -> tuple[str, str]:
    """Fail closed at every Muon checkpoint entry on the pinned private APIs."""
    try:
        mcore_version = importlib.metadata.version("megatron-core")
        eo_version = importlib.metadata.version("emerging-optimizers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Muon checkpoint requires megatron-core 0.17.0 and "
            "emerging-optimizers 0.3.0"
        ) from error
    if mcore_version != _SUPPORTED_MEGATRON_CORE_VERSION:
        raise RuntimeError(
            f"Muon checkpoint requires megatron-core 0.17.0, found {mcore_version}"
        )
    if eo_version != _SUPPORTED_EMERGING_OPTIMIZERS_VERSION:
        raise RuntimeError(
            f"Muon checkpoint requires emerging-optimizers 0.3.0, found {eo_version}"
        )
    return mcore_version, eo_version


@dataclass(frozen=True)
class GPUStagedMuonConfig:
    """GPU staging capacity for the CPU-resident Muon backend."""

    buffer_count: int = 2
    slot_size_mb: float = 128.0

    def __post_init__(self) -> None:
        if self.buffer_count < 1:
            raise ValueError("Muon staging buffer_count must be at least 1")
        if self.slot_size_mb <= 0:
            raise ValueError("Muon staging slot_size_mb must be positive")

    @property
    def slot_numel(self) -> int:
        return max(1, int(self.slot_size_mb * 1024 * 1024) // 4)


@dataclass
class MuonCPUSlabs:
    """Pinned FP32 master and momentum slabs authoritative between steps."""

    master: torch.Tensor
    momentum: torch.Tensor

    @classmethod
    def allocate(cls, numel: int) -> MuonCPUSlabs:
        if numel < 0:
            raise ValueError("Muon slab numel must be non-negative")
        kwargs = {"dtype": torch.float32, "device": "cpu", "pin_memory": True}
        return cls(
            master=torch.empty(numel, **kwargs),
            momentum=torch.zeros(numel, **kwargs),
        )


@dataclass(frozen=True)
class MuonOwnedUnit:
    """One complete owner-held matrix; Newton--Schulz never sees fragments."""

    param: torch.nn.Parameter
    group_index: int
    slab_offset: int
    numel: int


@dataclass(frozen=True)
class _FrozenOwnerParameter:
    """Immutable local authority for one layer-wise all-gather parameter."""

    param: torch.Tensor
    domain: str
    owner_rank: int
    ordinal: int
    shape: tuple[int, ...]
    numel: int
    stride: tuple[int, ...]
    layout: torch.layout
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    storage: Any
    storage_cdata: int
    param_data_ptr: int
    storage_data_ptr: int
    storage_nbytes: int
    storage_offset: int


def _freeze_owner_schema(
    domain: str,
    owner_lists: Sequence[Sequence[torch.Tensor]] | None,
) -> tuple[tuple[_FrozenOwnerParameter, ...], ...] | None:
    """Capture owner ordering and tensor/storage metadata at bind time."""
    if owner_lists is None:
        return None
    frozen: list[tuple[_FrozenOwnerParameter, ...]] = []
    ordinal = 0
    for owner_rank, params in enumerate(owner_lists):
        owner: list[_FrozenOwnerParameter] = []
        for param in params:
            storage = param.untyped_storage()
            owner.append(
                _FrozenOwnerParameter(
                    param=param,
                    domain=domain,
                    owner_rank=owner_rank,
                    ordinal=ordinal,
                    shape=tuple(param.shape),
                    numel=param.numel(),
                    stride=tuple(param.stride()),
                    layout=param.layout,
                    dtype=param.dtype,
                    device_type=param.device.type,
                    device_index=param.device.index,
                    storage=storage,
                    storage_cdata=int(storage._cdata),
                    # A zero pointer is a valid value for a zero-numel tensor;
                    # it is frozen and compared exactly, never used as a
                    # missing-authority sentinel.
                    param_data_ptr=int(param.data_ptr()),
                    storage_data_ptr=int(storage.data_ptr()),
                    storage_nbytes=int(storage.nbytes()),
                    storage_offset=param.storage_offset(),
                )
            )
            ordinal += 1
        frozen.append(tuple(owner))
    return tuple(frozen)


def _owner_schema_digest(
    domain: str,
    schema: tuple[tuple[_FrozenOwnerParameter, ...], ...] | None,
) -> tuple[int, int, int, int]:
    """Return a rank-stable structural digest; local storage identity is separate."""
    structural = (
        domain,
        None
        if schema is None
        else tuple(
            tuple(
                (
                    entry.owner_rank,
                    entry.ordinal,
                    entry.shape,
                    entry.numel,
                    entry.stride,
                    str(entry.layout),
                    str(entry.dtype),
                    entry.device_type,
                )
                for entry in owner
            )
            for owner in schema
        ),
    )
    digest = hashlib.sha256(repr(structural).encode()).digest()
    words = tuple(
        int.from_bytes(digest[offset : offset + 8], "little") & ((1 << 63) - 1)
        for offset in range(0, 32, 8)
    )
    if len(words) != 4:
        raise AssertionError("SHA-256 owner schema digest must contain four words")
    return words[0], words[1], words[2], words[3]


def _preflight_tp_gradient_participation(
    units: Sequence[MuonOwnedUnit],
    tp_groups: Mapping[str, Any] | None,
) -> tuple[bool, str | None]:
    """Run fixed-size raw-gradient votes in both official TP domains."""
    if tp_groups is None:
        return (
            any(
                getattr(unit.param, "main_grad", None) is not None
                or unit.param.grad is not None
                for unit in units
            ),
            None,
        )
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "staged Muon TP groups are bound but torch.distributed is not initialized"
        )

    any_active = False
    errors: list[str] = []
    for domain in ("dense", "expert"):
        group = tp_groups[domain]
        group_name = "expt_tp" if domain == "expert" else "tp"
        group_size, group_rank = _group_size_and_rank(group, name=group_name)
        params = tuple(
            unit.param
            for unit in units
            if ("expert" if getattr(unit.param, "expert_tp", False) else "dense")
            == domain
        )
        local_presence = tuple(
            getattr(param, "main_grad", None) is not None or param.grad is not None
            for param in params
        )
        local_active = any(local_presence)
        any_active |= local_active
        if group_size == 1:
            continue

        digest = hashlib.sha256(repr(local_presence).encode()).digest()
        digest_words = [
            int.from_bytes(digest[offset : offset + 8], "little") & ((1 << 63) - 1)
            for offset in range(0, 32, 8)
        ]
        vote = torch.tensor(
            [len(params), int(local_active), *digest_words],
            dtype=torch.int64,
            device=params[0].device
            if params
            else torch.device("cuda", torch.cuda.current_device()),
        )
        minimum = vote.clone()
        maximum = vote.clone()
        torch.distributed.all_reduce(
            minimum, op=torch.distributed.ReduceOp.MIN, group=group
        )
        torch.distributed.all_reduce(
            maximum, op=torch.distributed.ReduceOp.MAX, group=group
        )
        if not torch.equal(minimum, maximum):
            errors.append(
                "Muon TP peers have inconsistent gradient participation: "
                f"domain={domain}, group={group_name}, group_rank={group_rank}, "
                f"local={local_presence}"
            )
    return any_active, "; ".join(errors) if errors else None


class GPUStagedEmptyOptimizer:
    """State-free leaf used when one official partition owns no parameters."""

    manages_cpu_residency = True
    manages_master_weight = True

    def __init__(self, optimizer_kind: str) -> None:
        if optimizer_kind not in {"muon", "scalar_adamw"}:
            raise ValueError(f"unsupported empty optimizer kind: {optimizer_kind!r}")
        self.optimizer_kind = optimizer_kind
        self.param_groups: list[dict[str, Any]] = []
        self.state: dict[Any, Any] = {}
        self._checkpoint_load_error: BaseException | None = None

    @property
    def residency(self) -> str:
        return "CPU_RESIDENT"

    @property
    def cuda_state_numel(self) -> int:
        return 0

    @property
    def gpu_staging_state_numel(self) -> int:
        return 0

    @property
    def gpu_staging_numel(self) -> int:
        return 0

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if self._checkpoint_load_error is not None:
            raise RuntimeError(
                "cannot step after a failed checkpoint load; restart and recover"
            ) from self._checkpoint_load_error
        return closure() if closure is not None else None

    def bind_parallel_groups(self, *, tp: Any, expt_tp: Any) -> None:
        if tp is None or expt_tp is None:
            raise RuntimeError("staged Muon requires explicit tp and expt_tp groups")

    def preflight_step_activity(self) -> tuple[bool, str | None]:
        return False, None

    def drain(self) -> None:
        return None

    def offload_to_cpu(self) -> None:
        return None

    def restore_from_cpu(self) -> None:
        return None

    def prepare_checkpoint_save(self) -> None:
        if self._checkpoint_load_error is not None:
            raise RuntimeError(
                "cannot save after a failed checkpoint load; restart and recover"
            ) from self._checkpoint_load_error

    def begin_checkpoint_load(self) -> None:
        if self._checkpoint_load_error is not None:
            raise RuntimeError(
                "checkpoint load already failed; restart the process to recover"
            ) from self._checkpoint_load_error

    def complete_checkpoint_load(self) -> None:
        return None

    def mark_checkpoint_load_failed(self, error: BaseException) -> None:
        self._checkpoint_load_error = error

    def reset_from_model_params(self) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        self.prepare_checkpoint_save()
        return {"state": {}, "param_groups": []}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping) or state_dict != {
            "state": {},
            "param_groups": [],
        }:
            raise ValueError("empty staged optimizer checkpoint must contain no state")


class GPUStagedMuon(torch.optim.Optimizer):
    """Muon shell whose master and momentum live in pinned CPU slabs."""

    manages_cpu_residency = True
    manages_master_weight = True
    optimizer_kind = "muon"

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        staged_config: GPUStagedMuonConfig,
        orthogonalize: Callable[..., torch.Tensor],
        matmul_precision: Callable[[], AbstractContextManager[Any]],
        nesterov: bool,
        weight_decay_method: str,
        native_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        if weight_decay_method != "decoupled":
            raise ValueError("staged Muon MVP requires decoupled weight decay")
        super().__init__(params, defaults={})
        self.staged_config = staged_config
        self._orthogonalize = orthogonalize
        self._matmul_precision = matmul_precision
        self._nesterov = nesterov
        self._native_template = native_optimizer
        self._native_unit_optimizers: dict[int, torch.optim.Optimizer] = {}
        self.cpu_slabs: MuonCPUSlabs | None = None
        self._units: tuple[MuonOwnedUnit, ...] = ()
        self._runtime = StagedOptimizerRuntime(
            self.staged_config.buffer_count,
            ("master", "momentum", "grad", "workspace"),
        )
        self._tp_groups: dict[str, Any] | None = None
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
    def units(self) -> tuple[MuonOwnedUnit, ...]:
        return self._units

    @property
    def cuda_state_numel(self) -> int:
        return sum(
            value.numel()
            for state in self.state.values()
            for value in state.values()
            if isinstance(value, torch.Tensor) and value.is_cuda
        )

    @property
    def gpu_staging_state_numel(self) -> int:
        return sum(slot.master.numel() + slot.momentum.numel() for slot in self._slots)

    @property
    def gpu_staging_numel(self) -> int:
        """Total bounded slot storage, including grad and NS workspace."""
        return sum(
            slot.master.numel()
            + slot.momentum.numel()
            + slot.grad.numel()
            + slot.workspace.numel()
            for slot in self._slots
        )

    def bind_owned_params(
        self,
        param_groups: list[dict[str, Any]],
        *,
        empty_device: torch.device | None = None,
    ) -> None:
        """Bind only after official layer-wise ownership has been established."""
        if self._bound:
            raise RuntimeError("GPU-staged Muon is already bound")
        if len(param_groups) != len(self.param_groups):
            raise ValueError("Muon bound groups do not match official groups")
        self.param_groups = param_groups

        units: list[MuonOwnedUnit] = []
        devices: set[torch.device] = set()
        offset = 0
        seen: set[int] = set()
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                if id(param) in seen:
                    raise ValueError("duplicate parameter in Muon owner groups")
                seen.add(id(param))
                if param.ndim != 2:
                    raise ValueError(
                        "official Muon owner parameter must be a full 2D matrix"
                    )
                if not param.is_cuda:
                    raise ValueError("staged Muon parameters must be CUDA tensors")
                if param.numel() > self.staged_config.slot_numel:
                    raise ValueError(
                        "Muon slot is too small for an indivisible owner matrix: "
                        f"matrix_numel={param.numel()}, "
                        f"slot_numel={self.staged_config.slot_numel}"
                    )
                devices.add(param.device)
                units.append(MuonOwnedUnit(param, group_index, offset, param.numel()))
                offset += param.numel()
        if len(devices) > 1:
            raise ValueError("one staged Muon leaf cannot own multiple CUDA devices")
        if not devices and empty_device is None:
            raise ValueError("empty Muon owner requires an explicit CUDA device")
        device = next(iter(devices), empty_device)
        assert device is not None
        if device.type != "cuda":
            raise ValueError("Muon staging device must be CUDA")

        self._units = tuple(units)
        self.state.clear()
        if not units:
            self.cpu_slabs = None
            self._runtime.bind(capacity=None, device=device)
            self._bound = True
            return

        self.cpu_slabs = MuonCPUSlabs.allocate(offset)
        for unit in units:
            state = self.state[unit.param]
            state["master_param"] = self.cpu_slabs.master.narrow(
                0, unit.slab_offset, unit.numel
            ).view_as(unit.param)
            state["momentum_buffer"] = self.cpu_slabs.momentum.narrow(
                0, unit.slab_offset, unit.numel
            ).view_as(unit.param)

        self._runtime.bind(capacity=self.staged_config.slot_numel, device=device)
        self._bind_native_unit_optimizers()
        self._bound = True
        self._runtime.schedule_units(
            self._units,
            self._schedule_master_initialization,
            wait_for_compute=False,
        )
        self.drain()

    def bind_parallel_groups(self, *, tp: Any, expt_tp: Any) -> None:
        """Bind the official TP groups used by MCore's Muon implementation."""
        if self._bound:
            raise RuntimeError("Muon parallel groups must be bound before CPU state")
        if tp is None or expt_tp is None:
            raise RuntimeError("staged Muon requires explicit tp and expt_tp groups")
        self._tp_groups = {"dense": tp, "expert": expt_tp}

    def preflight_step_activity(self) -> tuple[bool, str | None]:
        """Validate raw TP gradient participation before MCore prepares gradients."""
        return _preflight_tp_gradient_participation(self._units, self._tp_groups)

    def _bind_native_unit_optimizers(self) -> None:
        """Bind native Muon steps to persistent views of reusable slot buffers."""
        if self._native_template is None:
            return
        for unit_index, unit in enumerate(self._units):
            slot = self._slots[unit_index % len(self._slots)]
            proxy = slot.master[: unit.numel].view_as(unit.param).detach()
            for name in (
                "expert_tp",
                "partition_dim",
                "partition_stride",
                "tensor_model_parallel",
                "is_qkv",
            ):
                if hasattr(unit.param, name):
                    setattr(proxy, name, getattr(unit.param, name))
            momentum = slot.momentum[: unit.numel].view_as(unit.param)
            native = copy.copy(self._native_template)
            vars(native).update(vars(self._native_template))
            group = dict(self.param_groups[unit.group_index])
            group["params"] = [proxy]
            native.param_groups = [group]
            native.state = {proxy: {"momentum_buffer": momentum}}
            self._native_unit_optimizers[id(unit.param)] = native

    def _schedule_master_initialization(
        self,
        unit: MuonOwnedUnit,
        slot_index: int,
        params_ready: torch.cuda.Event,
    ) -> None:
        assert self.cpu_slabs is not None
        slot = self._runtime.acquire_slot(slot_index)
        with torch.cuda.stream(slot.h2d_stream):
            slot.h2d_stream.wait_event(params_ready)
            slot.master[: unit.numel].copy_(unit.param.detach().view(-1))
            slot.h2d_done.record(slot.h2d_stream)
        with torch.cuda.stream(slot.d2h_stream):
            slot.d2h_stream.wait_event(slot.h2d_done)
            self.cpu_slabs.master.narrow(0, unit.slab_offset, unit.numel).copy_(
                slot.master[: unit.numel], non_blocking=True
            )
            slot.d2h_done.record(slot.d2h_stream)
        self._runtime.mark_d2h_pending(slot_index)

    def _schedule_update(
        self,
        unit: MuonOwnedUnit,
        slot_index: int,
        grads_ready: torch.cuda.Event,
    ) -> None:
        assert self.cpu_slabs is not None
        slot = self._runtime.acquire_slot(slot_index)
        slab_slice = slice(unit.slab_offset, unit.slab_offset + unit.numel)
        with torch.cuda.stream(slot.h2d_stream):
            slot.h2d_stream.wait_event(grads_ready)
            slot.master[: unit.numel].copy_(
                self.cpu_slabs.master[slab_slice], non_blocking=True
            )
            slot.momentum[: unit.numel].copy_(
                self.cpu_slabs.momentum[slab_slice], non_blocking=True
            )
            slot.h2d_done.record(slot.h2d_stream)

        group = self.param_groups[unit.group_index]
        lr = float(group["lr"])
        momentum_beta = float(group["momentum"])
        weight_decay = float(group["weight_decay"])
        with torch.cuda.stream(slot.compute_stream):
            slot.compute_stream.wait_event(grads_ready)
            slot.compute_stream.wait_event(slot.h2d_done)
            grad = getattr(unit.param, "decoupled_grad", None)
            if grad is None:
                grad = unit.param.grad
            if grad is not None:
                slot.grad[: unit.numel].copy_(grad.detach().view(-1))
                master = slot.master[: unit.numel].view_as(unit.param)
                momentum = slot.momentum[: unit.numel].view_as(unit.param)
                grad_matrix = slot.grad[: unit.numel].view_as(unit.param)
                native = self._native_unit_optimizers.get(id(unit.param))
                if native is not None:
                    proxy = native.param_groups[0]["params"][0]
                    proxy.grad = grad_matrix
                    native.step()
                    unit.param.copy_(proxy)
                else:
                    if weight_decay:
                        master.mul_(1.0 - lr * weight_decay)
                    momentum.lerp_(grad_matrix, 1.0 - momentum_beta)
                    update = (
                        grad_matrix.lerp(momentum, momentum_beta)
                        if self._nesterov
                        else momentum
                    )
                    with self._matmul_precision():
                        orthogonalized = self._orthogonalize(
                            unit.param,
                            update,
                            **{
                                key: value
                                for key, value in group.items()
                                if key != "params"
                            },
                        )
                    if orthogonalized.shape != unit.param.shape:
                        raise RuntimeError(
                            "official Muon orthogonalization changed matrix shape"
                        )
                    workspace = slot.workspace[: unit.numel].view_as(unit.param)
                    workspace.copy_(orthogonalized)
                    master.add_(workspace, alpha=-lr)
                    unit.param.copy_(master)
            slot.compute_done.record(slot.compute_stream)

        with torch.cuda.stream(slot.d2h_stream):
            slot.d2h_stream.wait_event(slot.compute_done)
            self.cpu_slabs.master[slab_slice].copy_(
                slot.master[: unit.numel], non_blocking=True
            )
            self.cpu_slabs.momentum[slab_slice].copy_(
                slot.momentum[: unit.numel], non_blocking=True
            )
            slot.d2h_done.record(slot.d2h_stream)
        self._runtime.mark_d2h_pending(slot_index)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if not self._bound:
            raise RuntimeError("bind_owned_params() must be called before Muon step")
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "cannot step after a failed checkpoint load; restart and recover"
            ) from self._runtime.checkpoint_load_error
        if closure is not None:
            raise ValueError("staged Muon does not support closures")
        if not self._units:
            self._runtime.residency = "CPU_RESIDENT"
            return None
        self._runtime.schedule_units(
            self._units, self._schedule_update, wait_for_compute=True
        )
        return None

    def drain(self) -> None:
        self._runtime.drain()

    def offload_to_cpu(self) -> None:
        self.drain()
        self._native_unit_optimizers.clear()
        self._runtime.release_slots()

    def restore_from_cpu(self) -> None:
        if not self._bound or not self._units or self._slots:
            return
        device = self._units[0].param.device
        self._runtime.restore_slots(
            capacity=self.staged_config.slot_numel, device=device
        )
        self._bind_native_unit_optimizers()

    def _validate_bound_state_views(self) -> None:
        if not self._bound:
            raise RuntimeError("GPU-staged Muon must be bound before checkpointing")
        if not self._units:
            if self.cpu_slabs is not None or self.state:
                raise RuntimeError("empty Muon ownership unexpectedly has state")
            return
        if self.cpu_slabs is None:
            raise RuntimeError("bound Muon optimizer has no CPU slabs")
        expected = {"master_param", "momentum_buffer"}
        for name, slab in (
            ("master", self.cpu_slabs.master),
            ("momentum", self.cpu_slabs.momentum),
        ):
            if slab.device.type != "cpu" or slab.dtype is not torch.float32:
                raise RuntimeError(f"Muon {name} slab must remain CPU FP32")
            if not slab.is_pinned():
                raise RuntimeError(f"Muon {name} slab lost pinned residency")
        for unit in self._units:
            state = self.state.get(unit.param)
            if not isinstance(state, Mapping) or set(state) != expected:
                raise RuntimeError("Muon checkpoint state schema changed after bind")
            for key, slab_name in (
                ("master_param", "master"),
                ("momentum_buffer", "momentum"),
            ):
                value = state[key]
                slab = getattr(self.cpu_slabs, slab_name)
                if (
                    value.device.type != "cpu"
                    or value.dtype is not torch.float32
                    or value.numel() != unit.numel
                    or value.untyped_storage().data_ptr()
                    != slab.untyped_storage().data_ptr()
                ):
                    raise RuntimeError(
                        f"Muon checkpoint state {key} lost its CPU slab alias"
                    )

    def prepare_checkpoint_save(self) -> None:
        """Drain staging before synchronous DCP reads authoritative CPU slabs."""
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "cannot save after a failed checkpoint load; restart and recover"
            ) from self._runtime.checkpoint_load_error
        self.drain()
        self._validate_bound_state_views()
        if self.cuda_state_numel != 0:
            raise RuntimeError("staged Muon checkpoint source contains CUDA state")

    def begin_checkpoint_load(self) -> None:
        """Prepare for one in-place synchronous load without rollback state."""
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "checkpoint load already failed; restart the process to recover"
            ) from self._runtime.checkpoint_load_error
        self.drain()
        self._validate_bound_state_views()

    def complete_checkpoint_load(self) -> None:
        """Validate CPU slab aliases after a successful load."""
        self.drain()
        self._validate_bound_state_views()

    def mark_checkpoint_load_failed(self, error: BaseException) -> None:
        """Fail-stop after an in-place load error; no rollback or retry."""
        self._runtime.mark_checkpoint_load_failed(error)

    def reset_from_model_params(self) -> None:
        """Rebuild masters and clear momentum after a model-only load."""
        if not self._bound:
            raise RuntimeError("GPU-staged Muon must be bound before state reset")
        if not self._units:
            return
        assert self.cpu_slabs is not None
        slots_were_released = not self._slots
        if slots_were_released:
            self.restore_from_cpu()
        try:
            self.drain()
            self._runtime.schedule_units(
                self._units,
                self._schedule_master_initialization,
                wait_for_compute=False,
            )
            self.drain()
            self.cpu_slabs.momentum.zero_()
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
                "cannot mutate after a failed checkpoint load; restart and recover"
            ) from self._runtime.checkpoint_load_error
        if key not in {"master_param", "momentum_buffer"}:
            raise KeyError(f"unsupported staged Muon state: {key}")
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
        self.prepare_checkpoint_save()
        return super().state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load CPU state in place without moving it to parameter devices."""
        if self._runtime.checkpoint_load_error is not None:
            raise RuntimeError(
                "cannot reload after a failed checkpoint load; restart and recover"
            ) from self._runtime.checkpoint_load_error
        if not isinstance(state_dict, Mapping) or set(state_dict) != {
            "state",
            "param_groups",
        }:
            raise ValueError("staged Muon checkpoint fields mismatch")
        loaded_groups = state_dict["param_groups"]
        if not isinstance(loaded_groups, (list, tuple)) or len(loaded_groups) != len(
            self.param_groups
        ):
            raise ValueError("staged Muon parameter groups mismatch")
        id_to_param: dict[int, torch.Tensor] = {}
        for current_group, loaded_group in zip(
            self.param_groups, loaded_groups, strict=True
        ):
            if not isinstance(loaded_group, Mapping) or set(loaded_group) != set(
                current_group
            ):
                raise ValueError("staged Muon parameter-group metadata mismatch")
            loaded_ids = loaded_group["params"]
            if not isinstance(loaded_ids, (list, tuple)) or len(loaded_ids) != len(
                current_group["params"]
            ):
                raise ValueError("staged Muon parameter-group ownership mismatch")
            for loaded_id, param in zip(
                loaded_ids, current_group["params"], strict=True
            ):
                if not isinstance(loaded_id, int) or isinstance(loaded_id, bool):
                    raise TypeError("staged Muon parameter ids must be integers")
                if loaded_id in id_to_param:
                    raise ValueError("staged Muon parameter id is duplicated")
                id_to_param[loaded_id] = param
        loaded_state = state_dict["state"]
        if not isinstance(loaded_state, Mapping) or set(loaded_state) != set(
            id_to_param
        ):
            raise ValueError("staged Muon state ownership mismatch")
        tensor_plan: list[tuple[torch.Tensor, str, torch.Tensor]] = []
        for loaded_id, values in loaded_state.items():
            if not isinstance(values, Mapping) or set(values) != {
                "master_param",
                "momentum_buffer",
            }:
                raise ValueError("staged Muon state schema mismatch")
            param = id_to_param[loaded_id]
            for key, value in values.items():
                destination = self.state[param][key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.dtype is not torch.float32
                    or value.shape != destination.shape
                ):
                    raise ValueError(f"loaded staged Muon {key} is incompatible")
                tensor_plan.append((param, key, value))
        self._validate_bound_state_views()
        for current_group, loaded_group in zip(
            self.param_groups, loaded_groups, strict=True
        ):
            params = current_group["params"]
            current_group.clear()
            current_group.update(
                {key: value for key, value in loaded_group.items() if key != "params"}
            )
            current_group["params"] = params
        for param, key, value in tensor_plan:
            self.state[param][key].copy_(value)


def _make_layerwise_leaf_class():
    from megatron.core.optimizer.optimizer import MegatronOptimizer

    class GPUStagedLayerWiseLeaf(MegatronOptimizer):
        """MCore leaf adapter around one staged Muon or scalar optimizer."""

        def __init__(self, optimizer: Any, config: Any, device: torch.device) -> None:
            super().__init__(optimizer, config, None)
            self.is_stub_optimizer = False
            self._scale = torch.tensor([1.0], dtype=torch.float32, device=device)
            self._step_activity: bool | None = None

        def _physical_parameters(self) -> list[torch.nn.Parameter]:
            return [
                param
                for group in self.optimizer.param_groups
                for param in group["params"]
            ]

        def preflight_step_activity(self) -> tuple[bool, str | None]:
            if getattr(self.optimizer, "optimizer_kind", None) == "muon":
                return self.optimizer.preflight_step_activity()
            return (
                any(
                    getattr(param, "main_grad", None) is not None
                    or param.grad is not None
                    for param in self._physical_parameters()
                ),
                None,
            )

        def set_step_activity(self, active: bool) -> None:
            if self._step_activity is not None:
                raise RuntimeError("staged Muon leaf already has an activity plan")
            self._step_activity = active

        def clear_step_activity(self) -> None:
            self._step_activity = None

        def get_parameters(self) -> list[torch.nn.Parameter]:
            if self._step_activity is False:
                return []
            return self._physical_parameters()

        def zero_grad(self, set_to_none: bool = True) -> None:
            for param in self.get_parameters():
                if set_to_none:
                    param.grad = None
                    param.decoupled_grad = None
                else:
                    if param.grad is not None:
                        param.grad.zero_()
                    if getattr(param, "decoupled_grad", None) is not None:
                        param.decoupled_grad.zero_()

        def get_loss_scale(self) -> torch.Tensor:
            return self._scale

        def prepare_grads(self) -> bool:
            for param in self._physical_parameters():
                grad = getattr(param, "main_grad", None)
                if grad is None:
                    grad = param.grad
                param.decoupled_grad = (
                    grad if self._step_activity is not False else None
                )
            return False

        def step_with_ready_grads(self) -> bool:
            if self._step_activity is False:
                return True
            self.optimizer.step()
            return True

        def step(self):
            return True, None, None

        def reload_model_params(self, state_dict=None) -> None:
            del state_dict
            raise RuntimeError(
                "Muon staged model-only optimizer reset is not supported in this MVP"
            )

        def state_dict(self):
            raise RuntimeError("Muon staged checkpoint is not supported in this MVP")

        def load_state_dict(self, state_dict):
            del state_dict
            raise RuntimeError("Muon staged checkpoint is not supported in this MVP")

        def sharded_state_dict(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("Muon staged checkpoint is not supported in this MVP")

        def offload_to_cpu(self) -> None:
            self.optimizer.offload_to_cpu()

        def restore_from_cpu(self) -> None:
            self.optimizer.restore_from_cpu()

    return GPUStagedLayerWiseLeaf


def _validate_mcore017_layerwise_allgather_contract(layerwise_cls: type[Any]) -> None:
    """Pin the private MCore method whose empty-shard bug we adapt."""
    method = layerwise_cls.allgather_params
    if tuple(inspect.signature(method).parameters) != ("self",):
        raise RuntimeError(
            "unsupported MCore 0.17 layer-wise allgather_params signature"
        )
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            "cannot verify MCore 0.17 layer-wise allgather_params source"
        ) from error
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != _MCORE017_LAYERWISE_ALLGATHER_SOURCE_SHA256:
        raise RuntimeError(
            "unsupported MCore 0.17 layer-wise allgather_params implementation: "
            f"sha256={digest}"
        )


@torch.no_grad()
def _allgather_mcore017_layerwise_params_empty_safe(self: Any) -> None:
    """MCore 0.17 all-gather that permits an empty first owner shard."""

    def _allgather(params_list: list[list[torch.Tensor]], group: Any) -> None:
        first_param = next(
            (param for owner_params in params_list for param in owner_params), None
        )
        if first_param is None:
            return
        rank = torch.distributed.get_rank(group)
        world_size = torch.distributed.get_world_size(group)
        if len(params_list) != world_size:
            raise RuntimeError(
                "Muon owner list does not match its process group: "
                f"owners={len(params_list)}, world_size={world_size}"
            )
        local_params = params_list[rank]
        source = (
            _flatten_dense_tensors(local_params)
            if local_params
            else torch.empty(
                0,
                device=first_param.device,
                dtype=first_param.dtype,
            )
        )
        flat_sizes = [sum(param.numel() for param in params) for params in params_list]
        gathered = [
            source
            if owner_rank == rank
            else torch.empty(
                size,
                device=first_param.device,
                dtype=first_param.dtype,
            )
            for owner_rank, size in enumerate(flat_sizes)
        ]
        torch.distributed.all_gather(gathered, source, group=group)
        for owner_rank, owner_params in enumerate(params_list):
            if owner_rank == rank or not owner_params:
                continue
            updated = _unflatten_dense_tensors(gathered[owner_rank], owner_params)
            for source_param, model_param in zip(updated, owner_params, strict=True):
                model_param.data.copy_(source_param)

    if self.pg_collection is None:
        return
    if self.dp_cp_params_list:
        _allgather(self.dp_cp_params_list, self.pg_collection.dp_cp)
    if self.expt_dp_params_list:
        _allgather(self.expt_dp_params_list, self.pg_collection.expt_dp)


def get_megatron_optimizer_with_dist_muon(
    mcore_config: Any,
    model: list[Any],
    *,
    pg_collection: Any | None = None,
) -> Any:
    """Build MCore's native layer-wise Muon without mutating caller config."""
    version = importlib.metadata.version("megatron-core")
    if version != _SUPPORTED_MEGATRON_CORE_VERSION:
        raise RuntimeError(
            f"distributed Muon supports megatron-core 0.17.0 exactly, found {version}"
        )
    if mcore_config.optimizer != "dist_muon":
        raise ValueError("distributed Muon requires optimizer='dist_muon'")
    if mcore_config.use_distributed_optimizer:
        raise ValueError("distributed Muon does not use Megatron dist-opt")
    if mcore_config.muon_scalar_optimizer != "adam":
        raise ValueError("AReaL distributed Muon supports only scalar AdamW")
    if not mcore_config.bf16 or mcore_config.fp16:
        raise ValueError("distributed Muon requires BF16 without FP16 loss scaling")
    if mcore_config.use_precision_aware_optimizer:
        raise ValueError("distributed Muon does not support precision-aware optimizer")

    from megatron.core.optimizer.layer_wise_optimizer import (
        LayerWiseDistributedOptimizer,
    )
    from megatron.core.optimizer.muon import get_megatron_muon_optimizer

    _validate_mcore017_layerwise_allgather_contract(LayerWiseDistributedOptimizer)
    build_config = copy.copy(mcore_config)
    optimizer = get_megatron_muon_optimizer(
        build_config,
        model,
        config_overrides=None,
        use_gloo_process_groups=True,
        layer_wise_distributed_optimizer=True,
        pg_collection=pg_collection,
    )
    if type(optimizer) is not LayerWiseDistributedOptimizer:
        raise TypeError(
            "official Muon builder did not return LayerWiseDistributedOptimizer"
        )
    optimizer.allgather_params = (
        _allgather_mcore017_layerwise_params_empty_safe.__get__(
            optimizer, type(optimizer)
        )
    )
    return optimizer


def _checkpoint_group_identity(
    group: Any, *, name: str, singleton_global_rank: int
) -> dict[str, Any]:
    size, rank = _group_size_and_rank(group, name=name)
    if group is None:
        members = [singleton_global_rank]
    else:
        ranks_fn = getattr(torch.distributed, "get_process_group_ranks", None)
        if not callable(ranks_fn):
            raise RuntimeError(
                f"Muon checkpoint cannot enumerate the {name} process group"
            )
        members = list(ranks_fn(group))
    if len(members) != size or len(set(members)) != size:
        raise RuntimeError(
            f"Muon checkpoint {name} group membership is inconsistent: "
            f"size={size}, members={members}"
        )
    return {"size": size, "rank": rank, "members": members}


def _validate_muon_checkpoint_group_metadata(
    checkpoint: Any, runtime: Mapping[str, Any], *, location: str
) -> None:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{location} must be a mapping")
    expected_keys = set(runtime) - {"params"}
    if set(checkpoint) != expected_keys:
        raise ValueError(
            f"{location} fields mismatch: missing={sorted(expected_keys - set(checkpoint))}, "
            f"unexpected={sorted(set(checkpoint) - expected_keys)}"
        )
    for name, value in checkpoint.items():
        if name in {
            "is_expert_parallel",
            "is_decoupled_lr",
            "maximize",
            "capturable",
            "differentiable",
            "fused",
            "nesterov",
        }:
            if value is not None and type(value) is not bool:
                raise TypeError(f"{location} field {name} must be bool or None")
        elif name == "betas":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError(f"{location} field betas must be a pair")
            if any(
                isinstance(beta, bool)
                or not isinstance(beta, (int, float))
                or not math.isfinite(float(beta))
                or not 0.0 <= float(beta) < 1.0
                for beta in value
            ):
                raise ValueError(f"{location} field betas must be finite in [0, 1)")
        elif name == "step":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"{location} field step must be a non-negative int")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                raise ValueError(f"{location} field {name} must be finite")
            if (
                name
                in {
                    "lr",
                    "initial_lr",
                    "max_lr",
                    "min_lr",
                    "weight_decay",
                    "momentum",
                    "eps",
                }
                and float(value) < 0.0
            ):
                raise ValueError(f"{location} field {name} must be non-negative")
            if name == "momentum" and not float(value) < 1.0:
                raise ValueError(f"{location} field momentum must be less than 1")
            if name == "eps" and float(value) <= 0.0:
                raise ValueError(f"{location} field eps must be positive")


_MUON_CHECKPOINT_INVARIANT_GROUPS = ("tp", "expt_tp", "pp", "cp", "ep")
_MUON_CHECKPOINT_OWNERSHIP_GROUPS = ("dp", "dp_cp", "expt_dp")
_MUON_CHECKPOINT_GROUPS = (
    "dp",
    "dp_cp",
    "tp",
    "cp",
    "ep",
    "expt_tp",
    "expt_dp",
    "pp",
)


def _muon_checkpoint_integer(
    value: Any,
    *,
    location: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{location} must be an integer, got {value!r}")
    if value < minimum or (maximum is not None and value >= maximum):
        interval = f"[{minimum}, {maximum})" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{location} must be in {interval}, got {value!r}")
    return value


def _validate_muon_checkpoint_cartesian_topology(
    participants: Sequence[Mapping[str, Any]],
    *,
    group_sizes: Mapping[str, int],
) -> None:
    """Prove that metadata is exactly MCore 0.17's two WORLD rank spaces."""
    from megatron.core.parallel_state import RankGenerator

    world_size = len(participants)
    order = "tp-cp-ep-dp-pp"
    dense_dimensions = ("tp", "pp", "cp", "dp")
    expert_dimensions = ("expt_tp", "pp", "ep", "expt_dp")

    def validate_coordinates(dimensions: tuple[str, ...], *, label: str) -> None:
        product = math.prod(group_sizes[name] for name in dimensions)
        if product != world_size:
            raise ValueError(
                f"Muon checkpoint {label} topology is not a WORLD Cartesian "
                f"product: dimensions={dimensions!r}, product={product}, "
                f"world_size={world_size}"
            )
        actual = {
            tuple(participant["groups"][name]["rank"] for name in dimensions)
            for participant in participants
        }
        expected = set(
            itertools.product(*(range(group_sizes[name]) for name in dimensions))
        )
        if len(actual) != world_size or actual != expected:
            raise ValueError(
                f"Muon checkpoint {label} coordinates do not cover the WORLD "
                "Cartesian product exactly: "
                f"duplicates={world_size - len(actual)}, "
                f"missing={sorted(expected - actual)!r}, "
                f"unexpected={sorted(actual - expected)!r}"
            )

    validate_coordinates(dense_dimensions, label="dense")
    validate_coordinates(expert_dimensions, label="expert")
    if group_sizes["dp_cp"] != group_sizes["dp"] * group_sizes["cp"]:
        raise ValueError(
            "Muon checkpoint dp_cp size does not match MCore's derived DP x CP "
            f"layout: actual={group_sizes['dp_cp']}, "
            f"expected={group_sizes['dp'] * group_sizes['cp']}"
        )

    dense_generator = RankGenerator(
        tp=group_sizes["tp"],
        ep=1,
        dp=group_sizes["dp"],
        pp=group_sizes["pp"],
        cp=group_sizes["cp"],
        order=order,
    )
    expert_generator = RankGenerator(
        tp=group_sizes["expt_tp"],
        ep=group_sizes["ep"],
        dp=group_sizes["expt_dp"],
        pp=group_sizes["pp"],
        cp=1,
        order=order,
    )
    if dense_generator.world_size != world_size:
        raise ValueError("Muon checkpoint dense MCore topology does not cover WORLD")
    if expert_generator.world_size != world_size:
        raise ValueError("Muon checkpoint expert MCore topology does not cover WORLD")
    if dense_generator.get_ranks("pp") != expert_generator.get_ranks("pp"):
        raise ValueError("Muon checkpoint dense and expert pipeline topology disagree")

    actual_groups = {
        name: sorted(
            {
                tuple(participant["groups"][name]["members"])
                for participant in participants
            }
        )
        for name in _MUON_CHECKPOINT_GROUPS
    }
    expected_groups = {
        "tp": dense_generator.get_ranks("tp"),
        "cp": dense_generator.get_ranks("cp"),
        "dp": dense_generator.get_ranks("dp"),
        "dp_cp": dense_generator.get_ranks("dp-cp"),
        "pp": dense_generator.get_ranks("pp"),
        "expt_tp": expert_generator.get_ranks("tp"),
        "ep": expert_generator.get_ranks("ep"),
        "expt_dp": expert_generator.get_ranks("dp"),
    }
    for name, expected in expected_groups.items():
        normalized_expected = sorted(tuple(members) for members in expected)
        if actual_groups[name] != normalized_expected:
            raise ValueError(
                f"Muon checkpoint {name} topology does not match MCore 0.17 "
                f"rank layout: expected={normalized_expected!r}, "
                f"actual={actual_groups[name]!r}"
            )


def _validate_muon_checkpoint_participant_topologies(
    topologies: Sequence[Any],
    *,
    trusted_global_ranks: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Validate rank authority and every process-group WORLD partition.

    ``trusted_global_ranks`` comes from the receive slots of the explicit
    WORLD-sized checkpoint control group.  Metadata's self-reported rank is
    never used to establish contributor authority.
    """
    participant_count = len(topologies)
    if participant_count == 0:
        raise ValueError("Muon checkpoint metadata participants are empty")
    if len(trusted_global_ranks) != participant_count:
        raise ValueError(
            "Muon checkpoint trusted participant count mismatch: "
            f"metadata={participant_count}, trusted={len(trusted_global_ranks)}"
        )
    trusted = [
        _muon_checkpoint_integer(
            rank,
            location=f"Muon checkpoint trusted participant slot {slot} rank",
            maximum=participant_count,
        )
        for slot, rank in enumerate(trusted_global_ranks)
    ]
    if set(trusted) != set(range(participant_count)):
        raise ValueError(
            "Muon checkpoint trusted participants must cover every WORLD rank "
            f"exactly once: actual={trusted!r}"
        )

    canonical: list[dict[str, Any] | None] = [None] * participant_count
    for slot, (trusted_rank, topology) in enumerate(
        zip(trusted, topologies, strict=True)
    ):
        location = (
            f"Muon checkpoint participant slot {slot} (rank {trusted_rank}) topology"
        )
        if not isinstance(topology, Mapping) or set(topology) != {
            "world_size",
            "global_rank",
            "groups",
        }:
            raise ValueError(f"{location} fields mismatch")
        world_size = _muon_checkpoint_integer(
            topology["world_size"],
            location=f"{location}.world_size",
            minimum=1,
        )
        if world_size != participant_count:
            raise ValueError(
                f"{location}.world_size mismatch: declared={world_size}, "
                f"participants={participant_count}"
            )
        declared_rank = _muon_checkpoint_integer(
            topology["global_rank"],
            location=f"{location}.global_rank",
            maximum=participant_count,
        )
        if declared_rank != trusted_rank:
            raise ValueError(
                f"{location}.global_rank does not match its trusted contributor: "
                f"declared={declared_rank}, trusted={trusted_rank}"
            )
        groups = topology["groups"]
        if not isinstance(groups, Mapping) or set(groups) != set(
            _MUON_CHECKPOINT_GROUPS
        ):
            raise ValueError(f"{location}.groups fields mismatch")
        normalized_groups: dict[str, Any] = {}
        for group_name in _MUON_CHECKPOINT_GROUPS:
            identity = groups[group_name]
            group_location = f"{location}.groups.{group_name}"
            if not isinstance(identity, Mapping) or set(identity) != {
                "size",
                "rank",
                "members",
            }:
                raise ValueError(f"{group_location} fields mismatch")
            size = _muon_checkpoint_integer(
                identity["size"],
                location=f"{group_location}.size",
                minimum=1,
                maximum=participant_count + 1,
            )
            group_rank = _muon_checkpoint_integer(
                identity["rank"],
                location=f"{group_location}.rank",
                maximum=size,
            )
            members_value = identity["members"]
            if not isinstance(members_value, list):
                raise ValueError(f"{group_location}.members must be a list")
            members = [
                _muon_checkpoint_integer(
                    member,
                    location=f"{group_location}.members[{index}]",
                    maximum=participant_count,
                )
                for index, member in enumerate(members_value)
            ]
            if len(members) != size or len(set(members)) != size:
                raise ValueError(
                    f"{group_location} membership mismatch: size={size}, "
                    f"members={members!r}"
                )
            if members != sorted(members):
                raise ValueError(
                    f"{group_location}.members do not use MCore's global-rank "
                    f"ordering: members={members!r}"
                )
            if trusted_rank not in members:
                raise ValueError(
                    f"{group_location} does not contain participant rank {trusted_rank}"
                )
            if members[group_rank] != trusted_rank:
                raise ValueError(
                    f"{group_location}.rank has the wrong member index: "
                    f"rank={group_rank}, members={members!r}, "
                    f"participant={trusted_rank}"
                )
            normalized_groups[group_name] = {
                "size": size,
                "rank": group_rank,
                "members": members,
            }
        canonical[trusted_rank] = {
            "world_size": participant_count,
            "global_rank": trusted_rank,
            "groups": normalized_groups,
        }

    if any(topology is None for topology in canonical):
        raise ValueError(
            "Muon checkpoint participant WORLD rank coverage is incomplete"
        )
    typed_canonical = [topology for topology in canonical if topology is not None]
    group_sizes: dict[str, int] = {}
    world = set(range(participant_count))
    for group_name in _MUON_CHECKPOINT_GROUPS:
        declarations: dict[tuple[int, ...], set[int]] = {}
        sizes: set[int] = set()
        for participant_rank, topology in enumerate(typed_canonical):
            identity = topology["groups"][group_name]
            members = tuple(identity["members"])
            declarations.setdefault(members, set()).add(participant_rank)
            sizes.add(identity["size"])
        if len(sizes) != 1:
            raise ValueError(
                f"Muon checkpoint {group_name} group sizes conflict across WORLD: "
                f"sizes={sorted(sizes)!r}"
            )
        covered: set[int] = set()
        for members, declaring_ranks in declarations.items():
            member_set = set(members)
            if declaring_ranks != member_set:
                raise ValueError(
                    f"Muon checkpoint {group_name} membership declarations are "
                    "inconsistent: "
                    f"members={members!r}, declared_by={sorted(declaring_ranks)!r}"
                )
            overlap = covered & member_set
            if overlap:
                raise ValueError(
                    f"Muon checkpoint {group_name} groups overlap on WORLD ranks "
                    f"{sorted(overlap)!r}"
                )
            covered.update(member_set)
        if covered != world:
            raise ValueError(
                f"Muon checkpoint {group_name} groups do not partition WORLD: "
                f"missing={sorted(world - covered)!r}, "
                f"extra={sorted(covered - world)!r}"
            )
        group_sizes[group_name] = sizes.pop()
    _validate_muon_checkpoint_cartesian_topology(
        typed_canonical, group_sizes=group_sizes
    )
    return typed_canonical, group_sizes


def _muon_checkpoint_coordinate(
    topology: Mapping[str, Any], *, domain: str
) -> dict[str, int]:
    groups = topology["groups"]
    coordinate = {
        "pp": groups["pp"]["rank"],
    }
    if domain == "dense":
        coordinate["tp"] = groups["tp"]["rank"]
    elif domain == "expert":
        coordinate["ep"] = groups["ep"]["rank"]
        coordinate["expt_tp"] = groups["expt_tp"]["rank"]
    else:
        raise ValueError(f"unsupported Muon checkpoint domain: {domain!r}")
    return coordinate


def _normalize_muon_checkpoint_parameter_coordinate(
    coordinate: Any,
    *,
    domain: str,
    topology: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    if domain == "dense":
        fields = {"pp": "pp", "tp": "tp"}
    elif domain == "expert":
        fields = {"pp": "pp", "ep": "ep", "expt_tp": "expt_tp"}
    else:
        raise ValueError(f"Muon checkpoint parameter domain is invalid: {domain!r}")
    if not isinstance(coordinate, Mapping) or set(coordinate) != set(fields):
        raise ValueError(
            f"Muon checkpoint {domain} parameter coordinate fields mismatch"
        )
    normalized = {}
    for field, group_name in fields.items():
        maximum = (
            topology["groups"][group_name]["size"] if topology is not None else None
        )
        normalized[field] = _muon_checkpoint_integer(
            coordinate[field],
            location=f"Muon checkpoint {domain} parameter coordinate {field}",
            maximum=maximum,
        )
    return normalized


def _muon_checkpoint_parameter_identity(
    leaf_index: int, parameter: Mapping[str, Any]
) -> tuple[Any, ...]:
    domain = parameter.get("domain")
    coordinate = _normalize_muon_checkpoint_parameter_coordinate(
        parameter.get("coordinate"), domain=domain
    )
    return (
        leaf_index,
        parameter.get("name"),
        domain,
        tuple(sorted(coordinate.items())),
    )


def _muon_checkpoint_parameter_invariant(
    parameter: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "name",
        "domain",
        "coordinate",
        "shape",
        "dtype",
        "state_kinds",
        "source_owner",
    }
    actual = set(parameter) if isinstance(parameter, Mapping) else set()
    if not isinstance(parameter, Mapping) or actual != expected:
        raise ValueError(
            "Muon checkpoint parameter fields mismatch: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    if not isinstance(parameter["name"], str) or not parameter["name"]:
        raise ValueError("Muon checkpoint parameter name must be a non-empty string")
    domain = parameter["domain"]
    coordinate = _normalize_muon_checkpoint_parameter_coordinate(
        parameter["coordinate"], domain=domain
    )
    shape = parameter["shape"]
    if not isinstance(shape, list):
        raise ValueError("Muon checkpoint parameter shape must be a list")
    normalized_shape = [
        _muon_checkpoint_integer(
            dimension,
            location=f"Muon checkpoint parameter shape[{index}]",
        )
        for index, dimension in enumerate(shape)
    ]
    if not isinstance(parameter["dtype"], str):
        raise ValueError("Muon checkpoint parameter dtype must be a string")
    state_kinds = parameter["state_kinds"]
    if not isinstance(state_kinds, list) or any(
        not isinstance(state_kind, str) for state_kind in state_kinds
    ):
        raise ValueError("Muon checkpoint parameter state_kinds must be a string list")
    return {
        "name": parameter["name"],
        "domain": domain,
        "coordinate": coordinate,
        "shape": normalized_shape,
        "dtype": parameter["dtype"],
        "state_kinds": list(state_kinds),
    }


def _validate_muon_checkpoint_leaf_metadata(
    leaf: Any, *, leaf_index: int, participant_rank: int
) -> None:
    expected = {"tree_path", "kind", "parameters", "param_groups"}
    if not isinstance(leaf, Mapping) or set(leaf) != expected:
        raise ValueError(
            f"Muon checkpoint rank {participant_rank} leaf {leaf_index} fields mismatch"
        )
    tree_path = leaf["tree_path"]
    if not isinstance(tree_path, list):
        raise ValueError(
            f"Muon checkpoint rank {participant_rank} leaf {leaf_index} tree_path "
            "must be a list"
        )
    normalized_path = [
        _muon_checkpoint_integer(
            value,
            location=(
                f"Muon checkpoint rank {participant_rank} leaf {leaf_index} "
                f"tree_path[{path_index}]"
            ),
        )
        for path_index, value in enumerate(tree_path)
    ]
    if normalized_path != [leaf_index]:
        raise ValueError(
            f"Muon checkpoint rank {participant_rank} leaf {leaf_index} tree_path "
            f"mismatch: actual={normalized_path!r}"
        )
    if leaf["kind"] not in _MUON_CHECKPOINT_STATE_KINDS:
        raise ValueError(
            f"Muon checkpoint rank {participant_rank} leaf {leaf_index} kind is invalid"
        )
    if not isinstance(leaf["parameters"], list) or not isinstance(
        leaf["param_groups"], list
    ):
        raise ValueError(
            f"Muon checkpoint rank {participant_rank} leaf {leaf_index} containers "
            "must be lists"
        )


def _validate_muon_checkpoint_source_owner(
    source_owner: Any,
    *,
    domain: str,
    ownership_sizes: Mapping[str, int],
    world_size: int,
    contributor_rank: int | None = None,
    contributor_topology: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "global_rank",
        "owner_rank",
        "owner_ordinal",
        "group_index",
        "parameter_index",
        "unit_order",
    }
    if not isinstance(source_owner, Mapping) or set(source_owner) != expected:
        raise ValueError("Muon checkpoint source_owner fields mismatch")
    owner_group = "expt_dp" if domain == "expert" else "dp_cp"
    limits = {
        "global_rank": world_size,
        "owner_rank": ownership_sizes[owner_group],
    }
    for field in (
        "global_rank",
        "owner_rank",
        "owner_ordinal",
        "group_index",
        "parameter_index",
    ):
        value = source_owner[field]
        _muon_checkpoint_integer(
            value,
            location=f"Muon checkpoint source owner {field}",
            maximum=limits.get(field),
        )
    unit_order = source_owner["unit_order"]
    if unit_order is not None:
        _muon_checkpoint_integer(
            unit_order,
            location="Muon checkpoint source owner unit_order",
        )
    if contributor_rank is not None and source_owner["global_rank"] != contributor_rank:
        raise ValueError(
            "Muon checkpoint source owner global_rank does not match its trusted "
            f"payload contributor: owner={source_owner['global_rank']}, "
            f"contributor={contributor_rank}"
        )
    if contributor_topology is not None:
        expected_owner_rank = contributor_topology["groups"][owner_group]["rank"]
        if source_owner["owner_rank"] != expected_owner_rank:
            raise ValueError(
                f"Muon checkpoint {domain} source owner coordinate mismatch: "
                f"owner_rank={source_owner['owner_rank']}, "
                f"participant_{owner_group}_rank={expected_owner_rank}"
            )


def merge_muon_checkpoint_metadata(
    local_metadata: Sequence[Mapping[str, Any]],
    *,
    trusted_global_ranks: Sequence[int],
) -> dict[str, Any]:
    """Create one DP-independent schema from rank-local official ownership."""
    if not local_metadata:
        raise ValueError("Muon checkpoint metadata participants are empty")
    participant_topologies, group_sizes = (
        _validate_muon_checkpoint_participant_topologies(
            [
                metadata.get("topology") if isinstance(metadata, Mapping) else None
                for metadata in local_metadata
            ],
            trusted_global_ranks=trusted_global_ranks,
        )
    )
    trusted_by_slot = list(trusted_global_ranks)
    first = local_metadata[0]
    required = {
        "schema_version",
        "megatron_core_version",
        "emerging_optimizers_version",
        "topology",
        "algorithm",
        "leaf_tree",
    }
    if set(first) != required:
        raise ValueError("Muon checkpoint local metadata fields mismatch")
    schema_version = first["schema_version"]
    if schema_version != _MUON_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Muon checkpoint schema version: {schema_version!r}"
        )
    invariant_sizes = {
        name: group_sizes[name] for name in _MUON_CHECKPOINT_INVARIANT_GROUPS
    }
    ownership_sizes = {
        name: group_sizes[name] for name in _MUON_CHECKPOINT_OWNERSHIP_GROUPS
    }
    if not isinstance(first["leaf_tree"], list):
        raise ValueError("Muon checkpoint leaf_tree must be a list")
    leaf_count = len(first["leaf_tree"])
    for leaf_index, leaf in enumerate(first["leaf_tree"]):
        _validate_muon_checkpoint_leaf_metadata(
            leaf, leaf_index=leaf_index, participant_rank=trusted_by_slot[0]
        )
    merged_leaves = [
        {
            "tree_path": list(first["leaf_tree"][index]["tree_path"]),
            "kind": first["leaf_tree"][index]["kind"],
            "parameters": [],
            "param_groups": [],
        }
        for index in range(leaf_count)
    ]
    parameters: list[dict[tuple[Any, ...], dict[str, Any]]] = [
        {} for _ in range(leaf_count)
    ]
    groups: list[dict[int, dict[str, Any]]] = [{} for _ in range(leaf_count)]

    for participant_slot, metadata in enumerate(local_metadata):
        participant_rank = trusted_by_slot[participant_slot]
        participant_topology = participant_topologies[participant_rank]
        if set(metadata) != required:
            raise ValueError(
                f"Muon checkpoint rank {participant_rank} metadata fields mismatch"
            )
        for field in (
            "schema_version",
            "megatron_core_version",
            "emerging_optimizers_version",
            "algorithm",
        ):
            if metadata[field] != first[field]:
                raise ValueError(
                    f"Muon checkpoint rank {participant_rank} {field} conflict"
                )
        leaves = metadata["leaf_tree"]
        if not isinstance(leaves, list) or len(leaves) != leaf_count:
            raise ValueError("Muon checkpoint leaf tree length conflict")
        for leaf_index, leaf in enumerate(leaves):
            _validate_muon_checkpoint_leaf_metadata(
                leaf,
                leaf_index=leaf_index,
                participant_rank=participant_rank,
            )
            merged_leaf = merged_leaves[leaf_index]
            if (
                list(leaf["tree_path"]) != merged_leaf["tree_path"]
                or leaf["kind"] != merged_leaf["kind"]
            ):
                raise ValueError(
                    f"Muon checkpoint rank {participant_rank} leaf {leaf_index} conflict"
                )
            for group_index, group in enumerate(leaf["param_groups"]):
                normalized_group = copy.deepcopy(dict(group))
                previous_group = groups[leaf_index].setdefault(
                    group_index, normalized_group
                )
                if previous_group != normalized_group:
                    raise ValueError(
                        f"Muon checkpoint leaf {leaf_index} param-group "
                        f"{group_index} conflicts across owners"
                    )
            for parameter in leaf["parameters"]:
                invariant = _muon_checkpoint_parameter_invariant(parameter)
                domain = invariant["domain"]
                if domain not in {"dense", "expert"}:
                    raise ValueError(
                        f"Muon checkpoint parameter domain is invalid: {domain!r}"
                    )
                expected_coordinate = (
                    {"pp", "tp"} if domain == "dense" else {"pp", "ep", "expt_tp"}
                )
                if set(invariant["coordinate"]) != expected_coordinate:
                    raise ValueError(
                        "Muon checkpoint parameter coordinate fields mismatch"
                    )
                contributor_coordinate = _muon_checkpoint_coordinate(
                    participant_topology, domain=domain
                )
                if invariant["coordinate"] != contributor_coordinate:
                    raise ValueError(
                        f"Muon checkpoint {domain} logical coordinate does not match "
                        f"trusted participant rank {participant_rank}: "
                        f"declared={invariant['coordinate']!r}, "
                        f"expected={contributor_coordinate!r}"
                    )
                expected_states = list(
                    _MUON_CHECKPOINT_STATE_KINDS[merged_leaf["kind"]]
                )
                if invariant["state_kinds"] != expected_states:
                    raise ValueError(
                        "Muon checkpoint parameter state kinds mismatch: "
                        f"expected={expected_states!r}, "
                        f"actual={invariant['state_kinds']!r}"
                    )
                identity = _muon_checkpoint_parameter_identity(leaf_index, parameter)
                previous = parameters[leaf_index].get(identity)
                if previous is not None:
                    raise ValueError(
                        "Muon checkpoint logical parameter has duplicate owners: "
                        f"identity={identity!r}, first={previous['source_owner']!r}, "
                        f"duplicate={parameter['source_owner']!r}"
                    )
                source_owner = parameter["source_owner"]
                _validate_muon_checkpoint_source_owner(
                    source_owner,
                    domain=domain,
                    ownership_sizes=ownership_sizes,
                    world_size=len(local_metadata),
                    contributor_rank=participant_rank,
                    contributor_topology=participant_topology,
                )
                parameters[leaf_index][identity] = copy.deepcopy(dict(parameter))

    for leaf_index, merged_leaf in enumerate(merged_leaves):
        merged_leaf["parameters"] = [
            parameters[leaf_index][identity]
            for identity in sorted(parameters[leaf_index], key=repr)
        ]
        merged_leaf["param_groups"] = [
            groups[leaf_index][index] for index in sorted(groups[leaf_index])
        ]

    return {
        "schema_version": schema_version,
        "megatron_core_version": first["megatron_core_version"],
        "emerging_optimizers_version": first["emerging_optimizers_version"],
        "topology": {
            "invariant_group_sizes": invariant_sizes,
            "source_ownership_group_sizes": ownership_sizes,
            "source_world_size": len(local_metadata),
            "source_participant_groups": [
                copy.deepcopy(topology["groups"]) for topology in participant_topologies
            ],
        },
        "algorithm": copy.deepcopy(first["algorithm"]),
        "leaf_tree": merged_leaves,
    }


def _make_staged_layerwise_class():
    from megatron.core.optimizer.layer_wise_optimizer import (
        LayerWiseDistributedOptimizer,
    )
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    class GPUStagedLayerWiseDistributedOptimizer(LayerWiseDistributedOptimizer):
        """Layer-wise wrapper preserving official ownership and collectives."""

        manages_cpu_residency = True
        supports_non_distributed_checkpoint = True

        def __init__(self, official: Any, leaves: list[Any]) -> None:
            self.pg_collection = official.pg_collection
            self.dp_cp_params_list = official.dp_cp_params_list
            self.expt_dp_params_list = official.expt_dp_params_list
            self._staged_owner_schema = {
                "dense": _freeze_owner_schema("dense", self.dp_cp_params_list),
                "expert": _freeze_owner_schema("expert", self.expt_dp_params_list),
            }
            self._staged_leaves = tuple(leaves)
            self.async_allgather = False
            self._checkpoint_parameter_names: dict[torch.Tensor, str] | None = None
            self._checkpoint_algorithm: dict[str, Any] | None = None
            ChainedOptimizer.__init__(self, leaves)
            self._checkpoint_process_group: Any | None = None

        def bind_managed_checkpoint_process_group(self, group: Any) -> None:
            """Bind the explicit WORLD-sized Gloo group used by checkpoint phases."""
            if group is None:
                raise RuntimeError(
                    "Muon checkpoint requires an explicit checkpoint process group"
                )
            self._checkpoint_process_group = group

        def configure_managed_checkpoint_schema(
            self,
            parameter_names: Mapping[torch.Tensor, str],
            *,
            algorithm: Mapping[str, Any] | None = None,
        ) -> None:
            names: dict[torch.Tensor, str] = {}
            params_by_name: dict[str, torch.Tensor] = {}
            for leaf in self._staged_leaves:
                for group in leaf.optimizer.param_groups:
                    for param in group["params"]:
                        try:
                            stable_name = parameter_names[param]
                        except KeyError as error:
                            raise RuntimeError(
                                "Muon checkpoint parameter lacks a stable model name"
                            ) from error
                        previous = names.setdefault(param, stable_name)
                        if previous != stable_name:
                            raise RuntimeError(
                                "Muon checkpoint parameter has ambiguous model names"
                            )
                        other = params_by_name.setdefault(stable_name, param)
                        if other is not param:
                            raise RuntimeError(
                                "Muon checkpoint stable model name is not unique: "
                                f"{stable_name!r}"
                            )
            self._checkpoint_parameter_names = names
            if algorithm is not None:
                self._checkpoint_algorithm = dict(algorithm)

        def _checkpoint_topology(self) -> dict[str, Any]:
            if self._checkpoint_parameter_names is None:
                raise RuntimeError("Muon checkpoint schema was not configured")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                checkpoint_group = self._checkpoint_process_group
                if checkpoint_group is None:
                    raise RuntimeError(
                        "Muon checkpoint process group was not bound before topology "
                        "construction"
                    )
                ranks_fn = getattr(torch.distributed, "get_process_group_ranks", None)
                if not callable(ranks_fn):
                    raise RuntimeError(
                        "Muon checkpoint cannot enumerate its explicit checkpoint "
                        "process group"
                    )
                checkpoint_members = list(ranks_fn(checkpoint_group))
                checkpoint_rank = torch.distributed.get_rank(checkpoint_group)
                if len(checkpoint_members) != torch.distributed.get_world_size(
                    checkpoint_group
                ) or not 0 <= checkpoint_rank < len(checkpoint_members):
                    raise RuntimeError(
                        "Muon checkpoint process-group membership is inconsistent"
                    )
                global_rank = checkpoint_members[checkpoint_rank]
                world_size = len(checkpoint_members)
            else:
                global_rank = 0
                world_size = 1
            groups = {
                name: _checkpoint_group_identity(
                    getattr(self.pg_collection, name),
                    name=name,
                    singleton_global_rank=global_rank,
                )
                for name in (
                    "tp",
                    "expt_tp",
                    "dp",
                    "dp_cp",
                    "expt_dp",
                    "pp",
                    "cp",
                    "ep",
                )
            }
            return {
                "world_size": world_size,
                "global_rank": global_rank,
                "groups": groups,
            }

        def _checkpoint_owner_lookup(
            self,
        ) -> dict[torch.Tensor, tuple[str, int, int]]:
            result: dict[torch.Tensor, tuple[str, int, int]] = {}
            for domain in ("dense", "expert"):
                schema = self._staged_owner_schema[domain]
                if schema is None:
                    continue
                for owner in schema:
                    for entry in owner:
                        value = (domain, entry.owner_rank, entry.ordinal)
                        previous = result.setdefault(entry.param, value)
                        if previous != value:
                            raise RuntimeError(
                                "Muon checkpoint parameter occurs in multiple owner domains"
                            )
            # MCore represents size-one DP/CP ownership with ``None`` rather
            # than a one-entry owner list.  In that exact case every local
            # parameter is the official owner; this is not a fallback for a
            # missing distributed ownership schema.
            next_ordinal = {"dense": 0, "expert": 0}
            for leaf in self._staged_leaves:
                for group in leaf.optimizer.param_groups:
                    domain = (
                        "expert" if group.get("is_expert_parallel", False) else "dense"
                    )
                    owner_schema = self._staged_owner_schema[domain]
                    owner_group = (
                        self.pg_collection.expt_dp
                        if domain == "expert"
                        else self.pg_collection.dp_cp
                    )
                    group_size, _ = _group_size_and_rank(
                        owner_group,
                        name="expt_dp" if domain == "expert" else "dp_cp",
                    )
                    for param in group["params"]:
                        if param in result:
                            continue
                        if owner_schema is not None or group_size != 1:
                            raise RuntimeError(
                                f"Muon checkpoint {domain} ownership is incomplete"
                            )
                        ordinal = next_ordinal[domain]
                        result[param] = (domain, 0, ordinal)
                        next_ordinal[domain] = ordinal + 1
            return result

        def _checkpoint_local_metadata(self) -> dict[str, Any]:
            mcore_version, eo_version = _require_muon_checkpoint_versions()
            if self._checkpoint_parameter_names is None:
                raise RuntimeError("Muon checkpoint schema was not configured")
            if self._checkpoint_algorithm is None:
                raise RuntimeError(
                    "Muon checkpoint algorithm schema was not configured"
                )
            topology = self._checkpoint_topology()
            owner_lookup = self._checkpoint_owner_lookup()
            leaves: list[dict[str, Any]] = []
            for leaf_index, leaf in enumerate(self._staged_leaves):
                base = leaf.optimizer
                kind = base.optimizer_kind
                state_kinds = list(_MUON_CHECKPOINT_STATE_KINDS[kind])
                parameters = []
                if kind == "muon":
                    unit_order = {
                        unit.param: unit_index
                        for unit_index, unit in enumerate(base.units)
                    }
                else:
                    unit_order = {
                        param: parameter_index
                        for parameter_index, param in enumerate(
                            param
                            for group in base.param_groups
                            for param in group["params"]
                        )
                    }
                for group_index, group in enumerate(base.param_groups):
                    for parameter_index, param in enumerate(group["params"]):
                        try:
                            domain, owner_rank, owner_ordinal = owner_lookup[param]
                            name = self._checkpoint_parameter_names[param]
                        except KeyError as error:
                            raise RuntimeError(
                                "Muon checkpoint owner metadata is incomplete"
                            ) from error
                        parameters.append(
                            {
                                "name": name,
                                "domain": domain,
                                "coordinate": _muon_checkpoint_coordinate(
                                    topology, domain=domain
                                ),
                                "shape": list(param.shape),
                                "dtype": str(param.dtype),
                                "state_kinds": state_kinds,
                                "source_owner": {
                                    "global_rank": topology["global_rank"],
                                    "owner_rank": owner_rank,
                                    "owner_ordinal": owner_ordinal,
                                    "group_index": group_index,
                                    "parameter_index": parameter_index,
                                    "unit_order": unit_order.get(param),
                                },
                            }
                        )
                leaves.append(
                    {
                        "tree_path": [leaf_index],
                        "kind": kind,
                        "parameters": parameters,
                        "param_groups": [
                            {
                                key: value
                                for key, value in group.items()
                                if key != "params"
                            }
                            for group in base.param_groups
                        ],
                    }
                )
            return {
                "schema_version": _MUON_CHECKPOINT_SCHEMA_VERSION,
                "megatron_core_version": mcore_version,
                "emerging_optimizers_version": eo_version,
                "topology": topology,
                "algorithm": self._checkpoint_algorithm,
                "leaf_tree": leaves,
            }

        def _checkpoint_metadata(self) -> dict[str, Any]:
            if not (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            ):
                return merge_muon_checkpoint_metadata(
                    [self._checkpoint_local_metadata()], trusted_global_ranks=[0]
                )
            group = self._checkpoint_process_group
            if group is None:
                raise RuntimeError(
                    "Muon checkpoint process group was not bound before metadata gather"
                )
            try:
                local_payload = {
                    "error": None,
                    "metadata": self._checkpoint_local_metadata(),
                }
            except BaseException as error:
                local_payload = {
                    "error": f"{type(error).__name__}: {error}",
                    "metadata": None,
                }
            participants: list[Any] = [
                None for _ in range(torch.distributed.get_world_size(group))
            ]
            torch.distributed.all_gather_object(
                participants, local_payload, group=group
            )
            failures = [
                (rank, payload["error"])
                for rank, payload in enumerate(participants)
                if payload["error"] is not None
            ]
            if failures:
                raise RuntimeError(
                    f"Muon checkpoint local metadata validation failed: {failures!r}"
                )
            ranks_fn = getattr(torch.distributed, "get_process_group_ranks", None)
            if not callable(ranks_fn):
                raise RuntimeError(
                    "Muon checkpoint cannot bind metadata participants to explicit "
                    "checkpoint process-group ranks"
                )
            trusted_global_ranks = list(ranks_fn(group))
            return merge_muon_checkpoint_metadata(
                [payload["metadata"] for payload in participants],
                trusted_global_ranks=trusted_global_ranks,
            )

        def managed_checkpoint_identities(
            self, model_parameter_names: Mapping[torch.Tensor, str]
        ) -> dict[tuple[int, ...], dict[str, Any]]:
            self.configure_managed_checkpoint_schema(model_parameter_names)
            metadata = self._checkpoint_metadata()
            result = {}
            for leaf_index, leaf in enumerate(metadata["leaf_tree"]):
                result[(leaf_index,)] = {
                    "version": _MUON_CHECKPOINT_SCHEMA_VERSION,
                    "tree_path": [leaf_index],
                    "kind": leaf["kind"],
                    "invariant_group_sizes": metadata["topology"][
                        "invariant_group_sizes"
                    ],
                    "parameters": [
                        _muon_checkpoint_parameter_invariant(parameter)
                        for parameter in leaf["parameters"]
                    ],
                }
            return result

        def _checkpoint_rank_prefix(self) -> str:
            return _MUON_CHECKPOINT_PREFIX

        def managed_checkpoint_outer_template(self) -> dict[str, Any]:
            from megatron.core.dist_checkpointing.mapping import ShardedObject

            prefix = self._checkpoint_rank_prefix()
            return {
                "metadata": ShardedObject(
                    f"{prefix}.metadata",
                    None,
                    (1,),
                    (0,),
                    replica_id=self._checkpoint_topology()["global_rank"],
                )
            }

        def validate_managed_checkpoint_outer_state(self, state: Any) -> None:
            if not isinstance(state, Mapping) or set(state) != {"metadata"}:
                raise ValueError("Muon checkpoint outer fields must be {'metadata'}")
            checkpoint = state["metadata"]
            if not isinstance(checkpoint, Mapping):
                raise TypeError("Muon checkpoint metadata must be a mapping")
            runtime = self._checkpoint_metadata()
            if set(checkpoint) != set(runtime):
                raise ValueError("Muon checkpoint metadata fields mismatch")
            for field in (
                "schema_version",
                "megatron_core_version",
                "emerging_optimizers_version",
                "algorithm",
            ):
                if checkpoint[field] != runtime[field]:
                    raise ValueError(
                        f"Muon checkpoint {field} mismatch: "
                        f"expected={runtime[field]!r}, actual={checkpoint[field]!r}"
                    )
            checkpoint_topology = checkpoint["topology"]
            if not isinstance(checkpoint_topology, Mapping) or set(
                checkpoint_topology
            ) != {
                "invariant_group_sizes",
                "source_ownership_group_sizes",
                "source_world_size",
                "source_participant_groups",
            }:
                raise ValueError("Muon checkpoint topology fields mismatch")
            if (
                checkpoint_topology["invariant_group_sizes"]
                != runtime["topology"]["invariant_group_sizes"]
            ):
                raise ValueError(
                    "Muon checkpoint invariant topology mismatch: "
                    f"expected={runtime['topology']['invariant_group_sizes']!r}, "
                    f"actual={checkpoint_topology['invariant_group_sizes']!r}"
                )
            invariant_sizes = checkpoint_topology["invariant_group_sizes"]
            if not isinstance(invariant_sizes, Mapping) or set(invariant_sizes) != set(
                _MUON_CHECKPOINT_INVARIANT_GROUPS
            ):
                raise ValueError("Muon checkpoint invariant topology fields mismatch")
            normalized_invariant_sizes = {
                name: _muon_checkpoint_integer(
                    invariant_sizes[name],
                    location=f"Muon checkpoint invariant topology {name} size",
                    minimum=1,
                )
                for name in _MUON_CHECKPOINT_INVARIANT_GROUPS
            }
            if (
                normalized_invariant_sizes
                != runtime["topology"]["invariant_group_sizes"]
            ):
                raise ValueError(
                    "Muon checkpoint invariant topology mismatch: "
                    f"expected={runtime['topology']['invariant_group_sizes']!r}, "
                    f"actual={normalized_invariant_sizes!r}"
                )
            source_sizes = checkpoint_topology["source_ownership_group_sizes"]
            if not isinstance(source_sizes, Mapping) or set(source_sizes) != set(
                _MUON_CHECKPOINT_OWNERSHIP_GROUPS
            ):
                raise ValueError(
                    "Muon checkpoint source ownership topology fields mismatch"
                )
            source_sizes = {
                name: _muon_checkpoint_integer(
                    source_sizes[name],
                    location=f"Muon checkpoint source ownership {name} size",
                    minimum=1,
                )
                for name in _MUON_CHECKPOINT_OWNERSHIP_GROUPS
            }
            source_world_size = checkpoint_topology["source_world_size"]
            source_world_size = _muon_checkpoint_integer(
                source_world_size,
                location="Muon checkpoint source world size",
                minimum=1,
            )
            source_participant_groups = checkpoint_topology["source_participant_groups"]
            if not isinstance(source_participant_groups, list):
                raise ValueError(
                    "Muon checkpoint source participant groups must be a list"
                )
            source_topologies, verified_source_sizes = (
                _validate_muon_checkpoint_participant_topologies(
                    [
                        {
                            "world_size": source_world_size,
                            "global_rank": global_rank,
                            "groups": participant_groups,
                        }
                        for global_rank, participant_groups in enumerate(
                            source_participant_groups
                        )
                    ],
                    trusted_global_ranks=list(range(source_world_size)),
                )
            )
            if {
                name: verified_source_sizes[name]
                for name in _MUON_CHECKPOINT_OWNERSHIP_GROUPS
            } != source_sizes:
                raise ValueError(
                    "Muon checkpoint source ownership group size metadata conflicts "
                    "with participant topology"
                )
            if {
                name: verified_source_sizes[name]
                for name in _MUON_CHECKPOINT_INVARIANT_GROUPS
            } != checkpoint_topology["invariant_group_sizes"]:
                raise ValueError(
                    "Muon checkpoint invariant group size metadata conflicts with "
                    "participant topology"
                )
            checkpoint_leaves = checkpoint["leaf_tree"]
            expected_leaves = runtime["leaf_tree"]
            if not isinstance(checkpoint_leaves, list) or len(checkpoint_leaves) != len(
                expected_leaves
            ):
                raise ValueError("Muon checkpoint leaf tree mismatch")
            for leaf_index, (loaded, runtime) in enumerate(
                zip(checkpoint_leaves, expected_leaves, strict=True)
            ):
                if set(loaded) != set(runtime):
                    raise ValueError(
                        f"Muon checkpoint leaf {leaf_index} fields mismatch"
                    )
                for field in ("tree_path", "kind"):
                    if loaded[field] != runtime[field]:
                        raise ValueError(
                            f"Muon checkpoint leaf {leaf_index} {field} mismatch"
                        )
                loaded_parameters = {
                    _muon_checkpoint_parameter_identity(
                        leaf_index, parameter
                    ): _muon_checkpoint_parameter_invariant(parameter)
                    for parameter in loaded["parameters"]
                }
                if len(loaded_parameters) != len(loaded["parameters"]):
                    raise ValueError(
                        f"Muon checkpoint leaf {leaf_index} has duplicate parameters"
                    )
                for parameter in loaded["parameters"]:
                    source_owner = parameter["source_owner"]
                    source_rank = _muon_checkpoint_integer(
                        source_owner.get("global_rank"),
                        location="Muon checkpoint source owner global_rank",
                        maximum=source_world_size,
                    )
                    _validate_muon_checkpoint_source_owner(
                        source_owner,
                        domain=parameter["domain"],
                        ownership_sizes=source_sizes,
                        world_size=source_world_size,
                        contributor_rank=source_rank,
                        contributor_topology=source_topologies[source_rank],
                    )
                    expected_coordinate = _muon_checkpoint_coordinate(
                        source_topologies[source_rank], domain=parameter["domain"]
                    )
                    if parameter["coordinate"] != expected_coordinate:
                        raise ValueError(
                            "Muon checkpoint logical parameter coordinate conflicts "
                            "with its source participant topology"
                        )
                runtime_parameters = {
                    _muon_checkpoint_parameter_identity(
                        leaf_index, parameter
                    ): _muon_checkpoint_parameter_invariant(parameter)
                    for parameter in runtime["parameters"]
                }
                if loaded_parameters != runtime_parameters:
                    raise ValueError(
                        f"Muon checkpoint leaf {leaf_index} parameter identity mismatch"
                    )
                groups = loaded["param_groups"]
                if not isinstance(groups, list) or len(groups) != len(
                    runtime["param_groups"]
                ):
                    raise ValueError(
                        f"Muon checkpoint leaf {leaf_index} param-group count mismatch"
                    )
                base = self._staged_leaves[leaf_index].optimizer
                if not base.param_groups:
                    if base.state:
                        raise RuntimeError(
                            f"Muon checkpoint empty leaf {leaf_index} retained state"
                        )
                    continue
                for group_index, (group, runtime_group) in enumerate(
                    zip(groups, base.param_groups, strict=True)
                ):
                    _validate_muon_checkpoint_group_metadata(
                        group,
                        runtime_group,
                        location=(
                            f"Muon checkpoint leaf {leaf_index} group {group_index}"
                        ),
                    )

        def _checkpoint_tensor_key(
            self,
            *,
            leaf_index: int,
            parameter_name: str,
            domain: str,
            coordinate: Mapping[str, int],
            state_kind: str,
        ) -> str:
            name_digest = hashlib.sha256(parameter_name.encode()).hexdigest()
            coordinate_digest = hashlib.sha256(
                json.dumps(
                    dict(coordinate), sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
            return (
                f"{_MUON_CHECKPOINT_PREFIX}.leaf_{leaf_index}.{domain}."
                f"coord_{coordinate_digest}."
                f"param_{name_digest}.{state_kind}"
            )

        def sharded_state_dict(
            self,
            model_sharded_state_dict: Any,
            is_loading: bool = False,
            **kwargs: Any,
        ) -> dict[str, Any]:
            del model_sharded_state_dict, is_loading, kwargs
            from megatron.core.dist_checkpointing.mapping import (
                ShardedObject,
                ShardedTensor,
            )

            _require_muon_checkpoint_versions()
            self.drain()
            metadata = self._checkpoint_metadata()
            local_metadata = self._checkpoint_local_metadata()
            prefix = self._checkpoint_rank_prefix()
            tensor_state: dict[str, Any] = {}
            for leaf_index, leaf in enumerate(self._staged_leaves):
                base = leaf.optimizer
                state_kinds = _MUON_CHECKPOINT_STATE_KINDS[base.optimizer_kind]
                descriptors = {
                    parameter["name"]: parameter
                    for parameter in local_metadata["leaf_tree"][leaf_index][
                        "parameters"
                    ]
                }
                for group in base.param_groups:
                    for param in group["params"]:
                        name = self._checkpoint_parameter_names[param]
                        descriptor = descriptors[name]
                        for state_kind in state_kinds:
                            value = base.state[param][state_kind]
                            flat_value = value.view(-1)
                            tensor_key = self._checkpoint_tensor_key(
                                leaf_index=leaf_index,
                                parameter_name=name,
                                domain=descriptor["domain"],
                                coordinate=descriptor["coordinate"],
                                state_kind=state_kind,
                            )
                            if tensor_key in tensor_state:
                                raise RuntimeError(
                                    "Muon checkpoint tensor key collision for "
                                    f"parameter {name!r} state {state_kind!r}"
                                )
                            tensor_state[tensor_key] = ShardedTensor(
                                tensor_key,
                                flat_value,
                                flat_value.dtype,
                                tuple(flat_value.shape),
                                tuple(flat_value.shape),
                                (0,),
                                (1,),
                                replica_id=0,
                            )
            return {
                "metadata": ShardedObject(
                    f"{prefix}.metadata",
                    metadata,
                    (1,),
                    (0,),
                    replica_id=self._checkpoint_topology()["global_rank"],
                ),
                "state": tensor_state,
            }

        def state_dict(self) -> dict[str, Any]:
            self.drain()
            metadata = self._checkpoint_metadata()
            local_metadata = self._checkpoint_local_metadata()
            tensor_state: dict[str, torch.Tensor] = {}
            for leaf_index, leaf in enumerate(self._staged_leaves):
                base = leaf.optimizer
                state_kinds = _MUON_CHECKPOINT_STATE_KINDS[base.optimizer_kind]
                descriptors = {
                    parameter["name"]: parameter
                    for parameter in local_metadata["leaf_tree"][leaf_index][
                        "parameters"
                    ]
                }
                for group in base.param_groups:
                    for param in group["params"]:
                        name = self._checkpoint_parameter_names[param]
                        descriptor = descriptors[name]
                        for state_kind in state_kinds:
                            tensor_key = self._checkpoint_tensor_key(
                                leaf_index=leaf_index,
                                parameter_name=name,
                                domain=descriptor["domain"],
                                coordinate=descriptor["coordinate"],
                                state_kind=state_kind,
                            )
                            if tensor_key in tensor_state:
                                raise RuntimeError(
                                    "Muon checkpoint tensor key collision for "
                                    f"parameter {name!r} state {state_kind!r}"
                                )
                            tensor_state[tensor_key] = base.state[param][
                                state_kind
                            ].view(-1)
            return {
                "metadata": metadata,
                "state": tensor_state,
            }

        def load_state_dict(self, state_dict: Any) -> None:
            if not isinstance(state_dict, Mapping):
                raise TypeError("Muon staged checkpoint must be a mapping")
            if "metadata" not in state_dict or set(state_dict) - {
                "metadata",
                "state",
            }:
                raise ValueError("Muon staged checkpoint fields mismatch")
            self.validate_managed_checkpoint_outer_state(
                {"metadata": state_dict["metadata"]}
            )
            # torch_dist omits an empty nested mapping from a rank-local load
            # result. Normalize only that container; exact local tensor keys
            # remain mandatory below for every owned parameter.
            tensor_state = state_dict.get("state", {})
            if not isinstance(tensor_state, Mapping):
                raise TypeError("Muon checkpoint tensor state must be a mapping")
            metadata_leaves = state_dict["metadata"]["leaf_tree"]
            local_metadata = self._checkpoint_local_metadata()
            apply_plan: list[
                tuple[
                    Any,
                    list[Mapping[str, Any]],
                    list[tuple[torch.Tensor, str, torch.Tensor]],
                ]
            ] = []
            expected_tensor_keys: set[str] = set()
            for leaf_index, (leaf, leaf_metadata) in enumerate(
                zip(self._staged_leaves, metadata_leaves, strict=True)
            ):
                base = leaf.optimizer
                groups = leaf_metadata["param_groups"]
                name_to_param = {
                    self._checkpoint_parameter_names[param]: param
                    for group in base.param_groups
                    for param in group["params"]
                }
                if len(name_to_param) != sum(
                    len(group["params"]) for group in base.param_groups
                ):
                    raise RuntimeError(
                        f"Muon checkpoint leaf {leaf_index} stable names are not unique"
                    )
                state_kinds = _MUON_CHECKPOINT_STATE_KINDS[base.optimizer_kind]
                descriptors = {
                    parameter["name"]: parameter
                    for parameter in local_metadata["leaf_tree"][leaf_index][
                        "parameters"
                    ]
                }
                tensor_plan: list[tuple[torch.Tensor, str, torch.Tensor]] = []
                for name, param in name_to_param.items():
                    descriptor = descriptors[name]
                    for state_kind in state_kinds:
                        tensor_key = self._checkpoint_tensor_key(
                            leaf_index=leaf_index,
                            parameter_name=name,
                            domain=descriptor["domain"],
                            coordinate=descriptor["coordinate"],
                            state_kind=state_kind,
                        )
                        expected_tensor_keys.add(tensor_key)
                        value = tensor_state.get(tensor_key)
                        if not isinstance(value, torch.Tensor):
                            raise TypeError(
                                f"Muon checkpoint tensor {tensor_key!r} must be a tensor"
                            )
                        if (
                            value.device.type != "cpu"
                            or value.dtype is not torch.float32
                        ):
                            raise TypeError(
                                f"Muon checkpoint parameter {name!r} {state_kind} "
                                "must be CPU FP32"
                            )
                        expected_shape = (param.numel(),)
                        if tuple(value.shape) != expected_shape:
                            raise ValueError(
                                f"Muon checkpoint parameter {name!r} {state_kind} "
                                f"shape mismatch: expected={expected_shape}, "
                                f"actual={tuple(value.shape)}"
                            )
                        tensor_plan.append((param, state_kind, value))
                apply_plan.append(
                    (base, groups if base.param_groups else [], tensor_plan)
                )

            if set(tensor_state) != expected_tensor_keys:
                raise ValueError(
                    "Muon checkpoint tensor key mismatch: "
                    f"missing={sorted(expected_tensor_keys - set(tensor_state))}, "
                    f"unexpected={sorted(set(tensor_state) - expected_tensor_keys)}"
                )

            # Every leaf, group, and tensor has passed validation before mutation.
            for base, groups, tensor_plan in apply_plan:
                for group, loaded_group in zip(base.param_groups, groups, strict=True):
                    params = group["params"]
                    group.clear()
                    group.update(loaded_group)
                    group["params"] = params
                for param, state_kind, value in tensor_plan:
                    base.set_scaled_state(param, state_kind, value.view_as(param))

        def _validate_owner_domain(
            self,
            *,
            domain: str,
            params_list: Sequence[Sequence[torch.Tensor]] | None,
            group: Any,
        ) -> tuple[list[int], torch.Tensor | None, int, str | None]:
            """Validate one frozen domain and complete its fixed subgroup vote."""
            group_size, group_rank = _group_size_and_rank(
                group,
                name="expt_dp" if domain == "expert" else "dp_cp",
            )
            expected = self._staged_owner_schema[domain]
            local_error: str | None = None
            flat_sizes: list[int] = []
            first_param: torch.Tensor | None = None
            try:
                if (params_list is None) != (expected is None):
                    raise RuntimeError(
                        f"staged Muon {domain} owner schema changed after bind"
                    )
                if params_list is None:
                    if expected is not None:
                        raise AssertionError("owner schema presence mismatch")
                elif not isinstance(params_list, Sequence) or isinstance(
                    params_list, (str, bytes)
                ):
                    raise RuntimeError(
                        f"staged Muon {domain} owner schema is not rank-indexed"
                    )
                elif expected is None or len(params_list) != group_size:
                    raise RuntimeError(
                        f"staged Muon {domain} owner schema length "
                        f"{len(params_list)} does not match group size {group_size}"
                    )
                elif len(expected) != len(params_list):
                    raise RuntimeError(
                        f"staged Muon {domain} owner schema changed after bind"
                    )
                else:
                    seen: set[int] = set()
                    expected_device: torch.device | None = None
                    expected_dtype: torch.dtype | None = None
                    for owner_rank, (params, expected_params) in enumerate(
                        zip(params_list, expected, strict=True)
                    ):
                        if not isinstance(params, Sequence) or isinstance(
                            params, (str, bytes)
                        ):
                            raise RuntimeError(
                                f"staged Muon {domain} owner rank {owner_rank} "
                                "entry is not a sequence"
                            )
                        if len(params) != len(expected_params):
                            raise RuntimeError(
                                f"staged Muon {domain} owner rank {owner_rank} "
                                "parameter order changed after bind"
                            )
                        size = 0
                        for param, frozen in zip(params, expected_params, strict=True):
                            if not isinstance(param, torch.Tensor):
                                raise RuntimeError(
                                    f"staged Muon {domain} owner rank "
                                    f"{owner_rank} contains a non-tensor"
                                )
                            if param is not frozen.param:
                                raise RuntimeError(
                                    f"staged Muon {domain} owner rank "
                                    f"{owner_rank} parameter order changed after bind"
                                )
                            if id(param) in seen:
                                raise RuntimeError(
                                    f"staged Muon {domain} owner schema contains a "
                                    f"duplicate parameter on owner rank {owner_rank}"
                                )
                            seen.add(id(param))
                            storage = param.untyped_storage()
                            metadata = (
                                ("shape", frozen.shape, tuple(param.shape)),
                                ("numel", frozen.numel, param.numel()),
                                ("stride", frozen.stride, tuple(param.stride())),
                                ("layout", frozen.layout, param.layout),
                                ("dtype", frozen.dtype, param.dtype),
                                ("device_type", frozen.device_type, param.device.type),
                                (
                                    "device_index",
                                    frozen.device_index,
                                    param.device.index,
                                ),
                                (
                                    "storage_object_identity",
                                    True,
                                    storage is frozen.storage,
                                ),
                                (
                                    "storage_cdata",
                                    frozen.storage_cdata,
                                    int(storage._cdata),
                                ),
                                (
                                    "storage_nbytes",
                                    frozen.storage_nbytes,
                                    int(storage.nbytes()),
                                ),
                                (
                                    "storage_data_ptr",
                                    frozen.storage_data_ptr,
                                    int(storage.data_ptr()),
                                ),
                                (
                                    "param_data_ptr",
                                    frozen.param_data_ptr,
                                    int(param.data_ptr()),
                                ),
                                (
                                    "storage_offset",
                                    frozen.storage_offset,
                                    param.storage_offset(),
                                ),
                            )
                            drift = [
                                f"{field}: expected={expected_value!r}, "
                                f"actual={actual_value!r}"
                                for field, expected_value, actual_value in metadata
                                if actual_value != expected_value
                            ]
                            if drift:
                                raise RuntimeError(
                                    f"staged Muon {domain} owner rank {owner_rank} "
                                    "parameter metadata changed after bind: "
                                    f"ordinal={frozen.ordinal}, fields=["
                                    f"{'; '.join(drift)}]"
                                )
                            if expected_device is None:
                                expected_device = param.device
                                expected_dtype = param.dtype
                            elif (
                                param.device != expected_device
                                or param.dtype != expected_dtype
                            ):
                                raise RuntimeError(
                                    f"staged Muon {domain} owner rank {owner_rank} "
                                    "has inconsistent parameter device or dtype"
                                )
                            first_param = param if first_param is None else first_param
                            size += param.numel()
                        flat_sizes.append(size)
            except BaseException as error:
                local_error = f"{type(error).__name__}: {error}"

            digest = _owner_schema_digest(domain, expected)
            if group_size > 1:
                vote_device = (
                    first_param.device
                    if first_param is not None and first_param.is_cuda
                    else torch.device("cuda", torch.cuda.current_device())
                )
                vote = torch.tensor(
                    [int(local_error is None), *digest],
                    dtype=torch.int64,
                    device=vote_device,
                )
                minimum = vote.clone()
                maximum = vote.clone()
                torch.distributed.all_reduce(
                    minimum, op=torch.distributed.ReduceOp.MIN, group=group
                )
                torch.distributed.all_reduce(
                    maximum, op=torch.distributed.ReduceOp.MAX, group=group
                )
                if not torch.equal(minimum, maximum) or minimum[0] == 0:
                    local_error = local_error or (
                        f"staged Muon {domain} owner metadata drifted on another rank"
                    )
            return flat_sizes, first_param, group_rank, local_error

        def _validate_owner_metadata(
            self,
        ) -> dict[str, tuple[list[int], torch.Tensor | None, int]]:
            """Validate both domains, then publish failures to the full chain."""
            if self.pg_collection is None:
                raise RuntimeError("staged Muon all-gather has no process groups")
            validated: dict[str, tuple[list[int], torch.Tensor | None, int]] = {}
            errors: list[str] = []
            for domain, params_list, group in (
                ("dense", self.dp_cp_params_list, self.pg_collection.dp_cp),
                ("expert", self.expt_dp_params_list, self.pg_collection.expt_dp),
            ):
                flat_sizes, first_param, group_rank, error = (
                    self._validate_owner_domain(
                        domain=domain, params_list=params_list, group=group
                    )
                )
                validated[domain] = flat_sizes, first_param, group_rank
                if error is not None:
                    errors.append(error)

            if torch.distributed.is_initialized():
                status = torch.tensor(
                    [int(not errors)],
                    dtype=torch.int64,
                    device=torch.cuda.current_device(),
                )
                torch.distributed.all_reduce(
                    status,
                    op=torch.distributed.ReduceOp.MIN,
                    group=torch.distributed.group.WORLD,
                )
                if status[0] == 0:
                    raise RuntimeError(
                        "; ".join(errors)
                        if errors
                        else "staged Muon owner metadata drifted on another rank"
                    )
            elif errors:
                raise RuntimeError("; ".join(errors))
            return validated

        @torch.no_grad()
        def allgather_params(self) -> None:
            """MCore 0.17 all-gather with empty owner shards handled safely."""
            validated = self._validate_owner_metadata()
            for domain, params_list, group in (
                ("dense", self.dp_cp_params_list, self.pg_collection.dp_cp),
                ("expert", self.expt_dp_params_list, self.pg_collection.expt_dp),
            ):
                flat_sizes, first_param, group_rank = validated[domain]
                if params_list is None or first_param is None or max(flat_sizes) == 0:
                    continue
                device = first_param.device
                dtype = first_param.dtype
                if device.type != "cuda":
                    raise RuntimeError(
                        f"staged Muon {domain} all-gather requires CUDA parameters"
                    )
                local_params = params_list[group_rank]
                src = (
                    _flatten_dense_tensors(local_params)
                    if local_params
                    else torch.empty(0, device=device, dtype=dtype)
                )
                if src.numel() != flat_sizes[group_rank]:
                    raise RuntimeError(
                        f"staged Muon {domain} local flat size does not match "
                        f"owner schema on rank {group_rank}"
                    )
                gather_list = [
                    src
                    if owner_rank == group_rank
                    else torch.empty(size, device=device, dtype=dtype)
                    for owner_rank, size in enumerate(flat_sizes)
                ]
                torch.distributed.all_gather(gather_list, src, group=group)
                for owner_rank, params in enumerate(params_list):
                    if owner_rank == group_rank or not params:
                        continue
                    updated = _unflatten_dense_tensors(gather_list[owner_rank], params)
                    for source, target in zip(updated, params, strict=True):
                        target.data.copy_(source)

        @torch.no_grad()
        def step(self):
            # This validation must precede ChainedOptimizer.prepare_grads(), the
            # WORLD norm/clip collectives, and every model/state mutation.
            self._validate_owner_metadata()
            activities: list[bool] = []
            errors: list[str] = []
            if tuple(self.chained_optimizers) != self._staged_leaves:
                errors.append("staged Muon optimizer leaf order changed after bind")
            for leaf_index, leaf in enumerate(self._staged_leaves):
                try:
                    active, error = leaf.preflight_step_activity()
                    activities.append(active)
                    if error is not None:
                        errors.append(f"leaf {leaf_index}: {error}")
                except BaseException as error:
                    activities.append(False)
                    errors.append(f"leaf {leaf_index}: {type(error).__name__}: {error}")

            chain_kinds = tuple(
                getattr(leaf.optimizer, "optimizer_kind", type(leaf).__name__)
                for leaf in self._staged_leaves
            )
            chain_digest = hashlib.sha256(repr(chain_kinds).encode()).digest()
            digest_words = [
                int.from_bytes(chain_digest[offset : offset + 8], "little")
                & ((1 << 63) - 1)
                for offset in range(0, 32, 8)
            ]
            world = torch.distributed.group.WORLD
            vote = torch.tensor(
                [int(not errors), int(any(activities)), *digest_words],
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            minimum = vote.clone()
            maximum = vote.clone()
            torch.distributed.all_reduce(
                minimum, op=torch.distributed.ReduceOp.MIN, group=world
            )
            torch.distributed.all_reduce(
                maximum, op=torch.distributed.ReduceOp.MAX, group=world
            )
            if not torch.equal(minimum[2:], maximum[2:]) or minimum[0] == 0:
                raise RuntimeError(
                    "; ".join(errors)
                    if errors
                    else "staged Muon step preflight failed on another rank"
                )

            planned: list[Any] = []
            try:
                for leaf, active in zip(self._staged_leaves, activities, strict=True):
                    leaf.set_step_activity(active)
                    planned.append(leaf)
                if maximum[1] == 0:
                    return (
                        True,
                        0.0,
                        0 if self.config.log_num_zeros_in_grad else None,
                    )
                return super().step()
            finally:
                self.drain()
                for leaf in reversed(planned):
                    leaf.clear_step_activity()

        def drain(self) -> None:
            for leaf in self.chained_optimizers:
                inner = getattr(leaf, "optimizer", None)
                if inner is not None:
                    inner.drain()

        @property
        def residency(self) -> str:
            states = {
                leaf.optimizer.residency
                for leaf in self.chained_optimizers
                if getattr(leaf, "optimizer", None) is not None
            }
            return "CPU_RESIDENT" if states <= {"CPU_RESIDENT"} else "STEP_ACTIVE"

        @property
        def cuda_state_numel(self) -> int:
            return sum(
                getattr(leaf.optimizer, "cuda_state_numel", 0)
                for leaf in self.chained_optimizers
                if getattr(leaf, "optimizer", None) is not None
            )

    return GPUStagedLayerWiseDistributedOptimizer


def _group_size_and_rank(group: Any, *, name: str) -> tuple[int, int]:
    if group is None:
        return 1, 0
    size_fn = getattr(group, "size", None)
    rank_fn = getattr(group, "rank", None)
    if not callable(size_fn) or not callable(rank_fn):
        raise RuntimeError(f"official {name} process group has no size/rank capability")
    size = size_fn()
    rank = rank_fn()
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise RuntimeError(f"official {name} process group has invalid size {size!r}")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < size:
        raise RuntimeError(
            f"official {name} process group rank {rank!r} is outside [0, {size})"
        )
    return size, rank


def _validate_owner_lists(
    owner_lists: Any,
    *,
    name: str,
    group_size: int,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    if not isinstance(owner_lists, Sequence) or isinstance(owner_lists, (str, bytes)):
        raise RuntimeError(
            f"official {name} owner list must be a rank-indexed sequence"
        )
    if len(owner_lists) != group_size:
        raise RuntimeError(
            f"official {name} owner list length {len(owner_lists)} does not match "
            f"group size {group_size}"
        )
    normalized = []
    for owner_rank, params in enumerate(owner_lists):
        if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
            raise RuntimeError(
                f"official {name} owner rank {owner_rank} entry must be a sequence"
            )
        if any(not isinstance(param, torch.Tensor) for param in params):
            raise RuntimeError(
                f"official {name} owner rank {owner_rank} contains a non-tensor"
            )
        normalized.append(tuple(params))
    return tuple(normalized)


def _parameter_identities(params: Iterable[torch.Tensor]) -> tuple[str, ...]:
    return tuple(
        f"id={id(param)},shape={tuple(param.shape)},dtype={param.dtype}"
        for param in params
    )


def _require_same_domain_params(
    *,
    domain: str,
    owner_rank: int,
    expected: Iterable[torch.Tensor],
    actual: Iterable[torch.Tensor],
    scope: str,
) -> None:
    expected_tuple = tuple(expected)
    actual_tuple = tuple(actual)
    expected_ids = {id(param) for param in expected_tuple}
    actual_ids = {id(param) for param in actual_tuple}
    if expected_ids != actual_ids:
        missing = tuple(
            param for param in expected_tuple if id(param) not in actual_ids
        )
        extra = tuple(param for param in actual_tuple if id(param) not in expected_ids)
        raise RuntimeError(
            f"official layer-wise {scope} ownership does not match owner lists: "
            f"domain={domain}, scope={scope}, owner_rank={owner_rank}, "
            f"expected={_parameter_identities(expected_tuple)}, "
            f"local={_parameter_identities(actual_tuple)}, "
            f"missing={_parameter_identities(missing)}, "
            f"extra={_parameter_identities(extra)}"
        )


def _validate_official_ownership(
    official: Any,
    expected_params: Iterable[torch.Tensor] | None = None,
) -> None:
    local_dense: list[torch.Tensor] = []
    local_expert: list[torch.Tensor] = []
    for leaf in official.chained_optimizers:
        for group in leaf.param_groups:
            is_expert_parallel = group.get("is_expert_parallel")
            if not isinstance(is_expert_parallel, bool):
                raise RuntimeError(
                    "official layer-wise param group has invalid ownership flag: "
                    f"is_expert_parallel={is_expert_parallel!r}"
                )
            target = local_expert if is_expert_parallel else local_dense
            target.extend(group["params"])
    local_params = local_dense + local_expert
    if len({id(param) for param in local_dense}) != len(local_dense):
        raise RuntimeError(
            "official layer-wise ownership contains duplicate local params: "
            "domain=dp_cp"
        )
    if len({id(param) for param in local_expert}) != len(local_expert):
        raise RuntimeError(
            "official layer-wise ownership contains duplicate local params: "
            "domain=expt_dp"
        )
    local_cross_domain = {id(param) for param in local_dense} & {
        id(param) for param in local_expert
    }
    if local_cross_domain:
        raise RuntimeError(
            "official layer-wise local parameter appears in both ownership domains: "
            f"parameter_ids={sorted(local_cross_domain)}"
        )
    pg = official.pg_collection
    if pg is None or not hasattr(pg, "dp_cp") or not hasattr(pg, "expt_dp"):
        raise RuntimeError("official layer-wise ownership has no process groups")
    dp_size, dp_rank = _group_size_and_rank(pg.dp_cp, name="dp_cp")
    expt_size, expt_rank = _group_size_and_rank(pg.expt_dp, name="expt_dp")

    expected_all = tuple(local_params if expected_params is None else expected_params)
    if len({id(param) for param in expected_all}) != len(expected_all):
        raise RuntimeError("expected model parameter ownership contains duplicates")
    expected_dense = tuple(
        param for param in expected_all if getattr(param, "allreduce", True)
    )
    expected_expert = tuple(
        param for param in expected_all if not getattr(param, "allreduce", True)
    )

    dp_owner_lists: tuple[tuple[torch.Tensor, ...], ...] | None = None
    if official.dp_cp_params_list is None:
        if dp_size != 1:
            raise RuntimeError(
                "official dp_cp owner list is missing for ownership group size "
                f"{dp_size}"
            )
    else:
        if dp_size == 1:
            raise RuntimeError(
                "official dp_cp owner list must be None for ownership group size 1"
            )
        dp_owner_lists = _validate_owner_lists(
            official.dp_cp_params_list,
            name="dp_cp",
            group_size=dp_size,
        )

    expt_owner_lists: tuple[tuple[torch.Tensor, ...], ...] | None = None
    expert_list_required = dp_size > 1 and expt_size > 1 and bool(expected_expert)
    if official.expt_dp_params_list is None:
        if expert_list_required:
            raise RuntimeError(
                "official expt_dp owner list is missing for non-empty expert ownership: "
                f"domain=expt_dp, owner_rank={expt_rank}, "
                f"expected={_parameter_identities(expected_expert)}"
            )
    else:
        if not expert_list_required:
            raise RuntimeError(
                "official expt_dp owner list must be None for MCore 0.17: "
                f"dp_cp_size={dp_size}, expt_dp_size={expt_size}, "
                f"expert_param_count={len(expected_expert)}"
            )
        expt_owner_lists = _validate_owner_lists(
            official.expt_dp_params_list,
            name="expt_dp",
            group_size=expt_size,
        )

    dense_owner_params = [param for params in dp_owner_lists or () for param in params]
    expert_owner_params = [
        param for params in expt_owner_lists or () for param in params
    ]
    if len({id(param) for param in dense_owner_params}) != len(dense_owner_params):
        raise RuntimeError(
            "official dp_cp owner lists contain duplicate params: domain=dp_cp"
        )
    if len({id(param) for param in expert_owner_params}) != len(expert_owner_params):
        raise RuntimeError(
            "official expt_dp owner lists contain duplicate params: domain=expt_dp"
        )
    cross_domain = {id(param) for param in dense_owner_params} & {
        id(param) for param in expert_owner_params
    }
    if cross_domain:
        raise RuntimeError(
            "official layer-wise parameter appears in both ownership domains: "
            f"parameter_ids={sorted(cross_domain)}"
        )

    dp_schema = tuple(dense_owner_params)
    if dp_owner_lists is None:
        dp_schema = tuple(local_dense)
    _require_same_domain_params(
        domain="dp_cp",
        owner_rank=dp_rank,
        expected=expected_dense,
        actual=dp_schema,
        scope="global",
    )

    expt_schema = tuple(expert_owner_params)
    if expt_owner_lists is None:
        expt_schema = tuple(local_expert)
    _require_same_domain_params(
        domain="expt_dp",
        owner_rank=expt_rank,
        expected=expected_expert,
        actual=expt_schema,
        scope="global",
    )

    expected_local_dense = tuple(
        dp_owner_lists[dp_rank] if dp_owner_lists is not None else expected_dense
    )
    expected_local_expert = tuple(
        expt_owner_lists[expt_rank] if expt_owner_lists is not None else expected_expert
    )
    _require_same_domain_params(
        domain="dp_cp",
        owner_rank=dp_rank,
        expected=expected_local_dense,
        actual=local_dense,
        scope="local",
    )
    _require_same_domain_params(
        domain="expt_dp",
        owner_rank=expt_rank,
        expected=expected_local_expert,
        actual=local_expert,
        scope="local",
    )


def _has_materialized_parameter_state(optimizer: Any) -> bool:
    """Accept empty defaultdict entries but reject any real parameter state."""
    for state in optimizer.state.values():
        if not isinstance(state, Mapping) or bool(state):
            return True
    return False


def _validate_muon_parallel_topology(
    official: Any,
    muon_optimizers: Sequence[Any],
    expected_params: Sequence[torch.Tensor],
    *,
    tp_mode: str,
) -> None:
    """Validate MCore's TP metadata and DP-owner collective coherence.

    MCore/EO own the actual Newton--Schulz implementation.  This preflight only
    proves that every TP peer will enter that implementation for the same
    owner-held TP-local parameter units and that expert parameters select the
    expert TP group.  It runs after official layer-wise sharding and before any
    staged CPU or CUDA storage is allocated.
    """
    pg = official.pg_collection
    required_groups = ("tp", "expt_tp", "dp_cp", "expt_dp")
    if pg is None:
        raise RuntimeError("official Muon topology has no process-group collection")
    groups: dict[str, Any] = {}
    group_metadata: dict[str, tuple[int, int]] = {}
    for name in required_groups:
        group = getattr(pg, name, None)
        if group is None:
            raise RuntimeError(
                f"official Muon topology is missing process group {name}"
            )
        groups[name] = group
        group_metadata[name] = _group_size_and_rank(group, name=name)

    expected_muon = tuple(
        param for param in expected_params if _is_mcore017_muon_parameter(param)
    )
    expected_ids = {id(param) for param in expected_muon}
    parameter_ordinals = {
        id(param): index for index, param in enumerate(expected_params)
    }
    local_by_dp_domain: dict[str, list[torch.Tensor]] = {"dense": [], "expert": []}
    local_by_tp_domain: dict[str, list[torch.Tensor]] = {"dense": [], "expert": []}
    for optimizer in muon_optimizers:
        if optimizer.pg_collection is not pg:
            raise RuntimeError(
                "official Muon optimizer and LayerWise wrapper use different "
                "process-group collections"
            )
        if optimizer.mode != tp_mode:
            raise RuntimeError(
                "official Muon TP mode does not match staged configuration: "
                f"official={optimizer.mode!r}, staged={tp_mode!r}"
            )
        for group_index, param_group in enumerate(optimizer.param_groups):
            is_expert = param_group.get("is_expert_parallel")
            if not isinstance(is_expert, bool):
                raise RuntimeError(
                    "official Muon param group has invalid expert ownership flag: "
                    f"group_index={group_index}, is_expert_parallel={is_expert!r}"
                )
            dp_domain = "expert" if is_expert else "dense"
            local_by_dp_domain[dp_domain].extend(param_group["params"])
            for param in param_group["params"]:
                tp_domain = "expert" if getattr(param, "expert_tp", False) else "dense"
                local_by_tp_domain[tp_domain].append(param)

    local_muon = local_by_dp_domain["dense"] + local_by_dp_domain["expert"]
    if any(id(param) not in expected_ids for param in local_muon):
        raise RuntimeError(
            "official Muon TP topology contains an unknown local parameter"
        )

    for param in expected_muon:
        expert_tp = getattr(param, "expert_tp", False)
        if not isinstance(expert_tp, bool):
            raise RuntimeError(
                "official Muon parameter has invalid expert_tp flag: "
                f"parameter={_parameter_identities((param,))}, "
                f"expert_tp={expert_tp!r}"
            )
        tp_group_name = "expt_tp" if expert_tp else "tp"
        tp_size, tp_rank = group_metadata[tp_group_name]
        tensor_parallel = getattr(param, "tensor_model_parallel", False)
        if not isinstance(tensor_parallel, bool):
            raise RuntimeError(
                "official Muon parameter has invalid tensor_model_parallel flag: "
                f"parameter={_parameter_identities((param,))}, "
                f"value={tensor_parallel!r}"
            )
        partition_dim = getattr(param, "partition_dim", -1)
        if isinstance(partition_dim, bool) or partition_dim not in {-1, 0, 1}:
            raise RuntimeError(
                "official Muon parameter has invalid TP partition_dim: "
                f"parameter={_parameter_identities((param,))}, "
                f"partition_dim={partition_dim!r}"
            )
        has_partition_axis = partition_dim in {0, 1}
        if tensor_parallel and not has_partition_axis:
            raise RuntimeError(
                "official Muon tensor-parallel metadata is inconsistent: "
                f"parameter={_parameter_identities((param,))}, "
                f"tensor_model_parallel={tensor_parallel}, "
                f"partition_dim={partition_dim}, group={tp_group_name}, "
                f"group_size={tp_size}, group_rank={tp_rank}"
            )
        if not tensor_parallel and has_partition_axis and not expert_tp and tp_size > 1:
            raise RuntimeError(
                "official dense Muon TP metadata has a partition axis without "
                "tensor_model_parallel: "
                f"parameter={_parameter_identities((param,))}, "
                f"partition_dim={partition_dim}, group_size={tp_size}, "
                f"group_rank={tp_rank}"
            )
        if has_partition_axis:
            partition_stride = getattr(param, "partition_stride", None)
            if (
                isinstance(partition_stride, bool)
                or not isinstance(partition_stride, int)
                or partition_stride < 1
            ):
                raise RuntimeError(
                    "official Muon parameter has invalid TP partition_stride: "
                    f"parameter={_parameter_identities((param,))}, "
                    f"partition_stride={partition_stride!r}"
                )
            if param.shape[partition_dim] < 1:
                raise RuntimeError(
                    "official Muon TP-local parameter shard is empty: "
                    f"parameter={_parameter_identities((param,))}, "
                    f"partition_dim={partition_dim}"
                )

    if not torch.distributed.is_initialized():
        return

    def _require_tp_owner_consensus(domain: str, group_name: str) -> None:
        group = groups[group_name]
        group_size, group_rank = group_metadata[group_name]
        signature = tuple(
            (
                parameter_ordinals[id(param)],
                tuple(param.shape),
                getattr(param, "partition_dim", -1),
            )
            for param in local_by_tp_domain[domain]
        )
        if group_size == 1:
            return
        gathered: list[Any] = [None] * group_size
        torch.distributed.all_gather_object(gathered, signature, group=group)
        if any(peer != gathered[0] for peer in gathered[1:]):
            raise RuntimeError(
                "official Muon TP peers selected different DP-owned units: "
                f"domain={domain}, group={group_name}, group_rank={group_rank}, "
                f"local={signature}, peers={gathered}"
            )

    _require_tp_owner_consensus("dense", "tp")
    _require_tp_owner_consensus("expert", "expt_tp")

    def _require_dp_schema_consensus(
        domain: str,
        group_name: str,
        owner_lists: Sequence[Sequence[torch.Tensor]] | None,
    ) -> None:
        group = groups[group_name]
        group_size, group_rank = group_metadata[group_name]
        if group_size == 1:
            return
        signature = (
            None
            if owner_lists is None
            else tuple(
                tuple(
                    (
                        parameter_ordinals[id(param)],
                        tuple(param.shape),
                        str(param.dtype),
                    )
                    for param in owner_params
                )
                for owner_params in owner_lists
            )
        )
        gathered: list[Any] = [None] * group_size
        torch.distributed.all_gather_object(gathered, signature, group=group)
        if any(peer != gathered[0] for peer in gathered[1:]):
            raise RuntimeError(
                "official Muon DP owner schemas differ across replicas: "
                f"domain={domain}, group={group_name}, group_rank={group_rank}, "
                f"local={signature}, peers={gathered}"
            )

    _require_dp_schema_consensus("dense", "dp_cp", official.dp_cp_params_list)
    _require_dp_schema_consensus("expert", "expt_dp", official.expt_dp_params_list)


def _is_mcore017_muon_parameter(param: torch.Tensor) -> bool:
    """Pinned copy of MCore 0.17's Muon predicate for its empty-Muon edge."""
    return (
        not getattr(param, "is_embedding_or_output_parameter", False)
        and param.ndim == 2
    )


def get_megatron_optimizer_with_gpu_staged_muon(
    mcore_config: Any,
    model: list[Any],
    staged_config: GPUStagedMuonConfig,
    *,
    pg_collection: Any | None = None,
) -> Any:
    """Run official classification/ownership, then bind staged rank-local state."""
    version = importlib.metadata.version("megatron-core")
    if version != _SUPPORTED_MEGATRON_CORE_VERSION:
        raise RuntimeError(
            f"GPU-staged Muon supports megatron-core 0.17.0 exactly, found {version}"
        )
    if mcore_config.use_distributed_optimizer:
        raise ValueError("staged Muon uses official layer-wise ownership, not dist-opt")
    if mcore_config.optimizer != "dist_muon":
        raise ValueError("staged Muon requires optimizer='dist_muon'")
    if mcore_config.muon_scalar_optimizer != "adam":
        raise ValueError("staged Muon requires AdamW for scalar parameters")
    if not mcore_config.decoupled_weight_decay:
        raise ValueError("staged Muon MVP requires decoupled weight decay")
    if not mcore_config.bf16 or mcore_config.fp16:
        raise ValueError("staged Muon MVP requires BF16 without FP16 loss scaling")
    if mcore_config.optimizer_cuda_graph:
        raise ValueError("staged Muon does not support optimizer CUDA graphs")
    if mcore_config.overlap_param_gather:
        raise ValueError("staged Muon MVP does not support overlap_param_gather")
    if getattr(mcore_config, "overlap_param_gather_with_optimizer_step", False):
        raise ValueError(
            "staged Muon MVP does not support overlap_param_gather_with_optimizer_step"
        )
    if mcore_config.use_precision_aware_optimizer:
        raise ValueError("staged Muon owns precision explicitly; PAO must be disabled")
    if mcore_config.muon_tp_mode != "duplicated":
        raise ValueError(
            "staged Muon currently supports only the verified duplicated TP mode; "
            f"found {mcore_config.muon_tp_mode!r}"
        )

    # This is deliberately before importing/calling either optimizer builder and
    # before touching ``model``.  Multiple staging streams have not yet been
    # validated against either dense-TP or expert-TP collective ordering.  An
    # explicit collection is authoritative; otherwise MCore's already-created
    # topology is queried without inspecting model chunks.
    from megatron.core.process_groups_config import ProcessGroupCollection

    topology_pg_collection = (
        pg_collection
        if pg_collection is not None
        else ProcessGroupCollection.use_mpu_process_groups(
            required_pgs=["tp", "expt_tp"]
        )
    )
    tp_size, _ = _group_size_and_rank(topology_pg_collection.tp, name="tp")
    expt_tp_size, _ = _group_size_and_rank(
        topology_pg_collection.expt_tp, name="expt_tp"
    )
    if (tp_size > 1 or expt_tp_size > 1) and staged_config.buffer_count != 1:
        raise ValueError(
            "staged Muon TP/expert-TP collectives require buffer_count=1; "
            f"tp_size={tp_size}, expt_tp_size={expt_tp_size}, "
            f"buffer_count={staged_config.buffer_count}"
        )

    from megatron.core.optimizer import (
        HAVE_EMERGING_OPTIMIZERS,
    )
    from megatron.core.optimizer import (
        get_megatron_optimizer as get_scalar_optimizer,
    )
    from megatron.core.optimizer.layer_wise_optimizer import (
        LayerWiseDistributedOptimizer,
    )
    from megatron.core.optimizer.muon import (
        TensorParallelMuon,
        get_megatron_muon_optimizer,
    )
    from megatron.core.optimizer.optimizer import FP32Optimizer

    if not HAVE_EMERGING_OPTIMIZERS:
        raise ImportError(
            "MCore 0.17 Muon requires its optional emerging-optimizers backend; "
            "the staged factory will not substitute a different algorithm"
        )
    eo_version = importlib.metadata.version("emerging-optimizers")
    if eo_version != _SUPPORTED_EMERGING_OPTIMIZERS_VERSION:
        raise RuntimeError(
            "staged Muon supports emerging-optimizers 0.3.0 exactly, "
            f"found {eo_version}"
        )
    builder_parameters = tuple(
        inspect.signature(get_megatron_muon_optimizer).parameters
    )
    if builder_parameters != (
        "config",
        "model_chunks",
        "config_overrides",
        "use_gloo_process_groups",
        "layer_wise_distributed_optimizer",
        "pg_collection",
    ):
        raise RuntimeError(
            f"unsupported MCore 0.17 Muon builder signature: {builder_parameters}"
        )
    if tuple(
        inspect.signature(LayerWiseDistributedOptimizer.shard_params).parameters
    ) != (
        "self",
        "optimizers",
    ):
        raise RuntimeError("unsupported MCore 0.17 layer-wise ownership signature")
    _validate_mcore017_layerwise_allgather_contract(LayerWiseDistributedOptimizer)

    trainable = []
    checkpoint_parameter_names: dict[torch.Tensor, str] = {}
    for model_index, chunk in enumerate(model):
        module = chunk.module if hasattr(chunk, "module") else chunk
        prefix = "model" if len(model) == 1 else f"model{model_index}"
        named_parameters = getattr(module, "named_parameters", None)
        if callable(named_parameters):
            parameters_with_names = named_parameters()
        else:
            parameters = getattr(module, "parameters", None)
            if not callable(parameters):
                raise TypeError("Muon model chunk has no parameter iterator")
            parameters_with_names = (
                (f"parameter_{index}", param)
                for index, param in enumerate(parameters())
            )
        for name, param in parameters_with_names:
            stable_name = f"{prefix}.{name}"
            previous = checkpoint_parameter_names.setdefault(param, stable_name)
            if previous != stable_name:
                raise RuntimeError(
                    "Muon checkpoint parameter has multiple stable model names: "
                    f"{previous!r} and {stable_name!r}"
                )
            if param.requires_grad:
                trainable.append(param)
    if not trainable:
        raise ValueError("staged Muon requires at least one trainable parameter")
    if any(not param.is_cuda for param in trainable):
        raise ValueError("staged Muon requires CUDA model parameters")
    if len({param.device for param in trainable}) != 1:
        raise ValueError("staged Muon requires one CUDA device per process")
    device = trainable[0].device

    build_config = copy.copy(mcore_config)
    build_config.bf16 = False
    build_config.fp16 = False
    build_config.use_distributed_optimizer = False
    build_config.use_precision_aware_optimizer = False
    build_config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = False
    build_config.main_grads_dtype = torch.float32
    build_config.main_params_dtype = torch.float32
    build_config.exp_avg_dtype = torch.float32
    build_config.exp_avg_sq_dtype = torch.float32
    build_config.muon_scalar_optimizer = "adam"
    build_config.overlap_param_gather = False
    build_config.overlap_param_gather_with_optimizer_step = False

    has_muon_parameters = any(_is_mcore017_muon_parameter(p) for p in trainable)
    if has_muon_parameters:
        official = get_megatron_muon_optimizer(
            build_config,
            model,
            use_gloo_process_groups=False,
            layer_wise_distributed_optimizer=True,
            pg_collection=pg_collection,
        )
    else:
        # MCore 0.17 constructs TensorParallelMuon before it checks whether the
        # official Muon class is empty, and torch.optim rejects an empty input.
        # For this exact edge, use MCore's normal Adam param-group builder and
        # the official layer-wise sharder.  The predicate above is the pinned
        # MCore 0.17 predicate and includes the embedding/output exclusion.
        resolved_pg_collection = (
            pg_collection
            if pg_collection is not None
            else ProcessGroupCollection.use_mpu_process_groups()
        )
        build_config.optimizer = "adam"
        scalar_chain = get_scalar_optimizer(
            build_config,
            model,
            use_gloo_process_groups=False,
            pg_collection=resolved_pg_collection,
        )
        official = LayerWiseDistributedOptimizer(
            list(scalar_chain.chained_optimizers),
            build_config,
            resolved_pg_collection,
            init_state_fn_list=None,
            model_chunks=None,
            async_allgather=False,
        )

    if type(official) is not LayerWiseDistributedOptimizer:
        raise TypeError(
            "official Muon builder did not return LayerWiseDistributedOptimizer"
        )
    if official.async_allgather:
        raise RuntimeError("staged Muon MVP requires synchronous official all-gather")
    _validate_official_ownership(official, trainable)

    official_bases = []
    for official_leaf in official.chained_optimizers:
        if type(official_leaf) is not FP32Optimizer:
            raise TypeError(
                "unsupported official Muon wrapper before staged ownership bind: "
                f"{type(official_leaf).__name__}"
            )
        base = official_leaf.optimizer
        if official_leaf.is_stub_optimizer:
            if base is not None or official_leaf.param_groups:
                raise RuntimeError("malformed official empty scalar optimizer leaf")
            official_bases.append(None)
            continue
        if base is None:
            raise RuntimeError("official non-stub Muon leaf has no base optimizer")
        if _has_materialized_parameter_state(base):
            raise RuntimeError("official Muon leaf allocated state before staged bind")
        official_bases.append(base)
    muon_leaf_count = sum(
        isinstance(base, TensorParallelMuon) for base in official_bases
    )
    expected_muon_leaf_count = 1 if has_muon_parameters else 0
    if muon_leaf_count != expected_muon_leaf_count:
        raise RuntimeError(
            "official Muon classification produced "
            f"{muon_leaf_count} Muon leaves, expected {expected_muon_leaf_count}"
        )
    _validate_muon_parallel_topology(
        official,
        tuple(base for base in official_bases if isinstance(base, TensorParallelMuon)),
        tuple(trainable),
        tp_mode=mcore_config.muon_tp_mode,
    )
    if not has_muon_parameters:
        official_bases.insert(0, _EMPTY_MUON_LEAF)

    # MCore's clip/norm helpers use this derived capability to read the FP32
    # ``decoupled_grad`` attached by our leaf wrapper.  Keep this on an
    # instance-local config copy so a rejected factory never mutates the
    # caller's configuration.
    staged_wrapper_config = copy.copy(mcore_config)
    staged_wrapper_config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True
    staged_wrapper_config.overlap_param_gather = False
    staged_wrapper_config.overlap_param_gather_with_optimizer_step = False
    leaf_cls = _make_layerwise_leaf_class()
    staged_leaves = []
    for base in official_bases:
        if base is _EMPTY_MUON_LEAF:
            staged_inner = GPUStagedEmptyOptimizer("muon")
            staged_inner.bind_parallel_groups(
                tp=official.pg_collection.tp,
                expt_tp=official.pg_collection.expt_tp,
            )
        elif isinstance(base, TensorParallelMuon):
            from emerging_optimizers.utils import fp32_matmul_precision

            staged_inner = GPUStagedMuon(
                base.param_groups,
                staged_config=staged_config,
                orthogonalize=base.orthogonalize,
                matmul_precision=lambda base=base: fp32_matmul_precision(
                    base.fp32_matmul_prec
                ),
                nesterov=base.nesterov,
                weight_decay_method=base.weight_decay_method,
                native_optimizer=base,
            )
            staged_inner.bind_parallel_groups(
                tp=official.pg_collection.tp,
                expt_tp=official.pg_collection.expt_tp,
            )
            staged_inner.bind_owned_params(base.param_groups, empty_device=device)
        elif base is None or not any(
            group.get("params") for group in base.param_groups
        ):
            staged_inner = GPUStagedEmptyOptimizer("scalar_adamw")
        else:
            staged_inner = GPUStagedAdamW(
                base.param_groups,
                lr=mcore_config.lr,
                betas=(mcore_config.adam_beta1, mcore_config.adam_beta2),
                eps=mcore_config.adam_eps,
                weight_decay=mcore_config.weight_decay,
                staged_config=GPUStagedAdamWConfig(
                    buffer_count=staged_config.buffer_count,
                    bucket_size_mb=staged_config.slot_size_mb,
                ),
                adam_w_mode=mcore_config.decoupled_weight_decay,
                master_weights=True,
                use_decoupled_grad=True,
            )
            staged_inner.bind_owned_params(
                base.param_groups,
                empty_device=device,
            )
            staged_inner.optimizer_kind = "scalar_adamw"
        staged_leaves.append(leaf_cls(staged_inner, staged_wrapper_config, device))

    staged_kinds = [leaf.optimizer.optimizer_kind for leaf in staged_leaves]
    if (
        not staged_kinds
        or staged_kinds[0] != "muon"
        or any(kind != "scalar_adamw" for kind in staged_kinds[1:])
    ):
        raise RuntimeError(
            "staged Muon chain topology must be [muon, scalar_adamw...], "
            f"found {staged_kinds}"
        )

    staged_layerwise_cls = _make_staged_layerwise_class()
    result = staged_layerwise_cls(official, staged_leaves)
    result.configure_managed_checkpoint_schema(
        checkpoint_parameter_names,
        algorithm={
            "momentum": mcore_config.muon_momentum,
            "use_nesterov": mcore_config.muon_use_nesterov,
            "fp32_matmul_prec": mcore_config.muon_fp32_matmul_prec,
            "coefficient_type": mcore_config.muon_coefficient_type,
            "num_ns_steps": mcore_config.muon_num_ns_steps,
            "scale_mode": mcore_config.muon_scale_mode,
            "split_qkv": mcore_config.muon_split_qkv,
            "tp_mode": mcore_config.muon_tp_mode,
            "extra_scale_factor": mcore_config.muon_extra_scale_factor,
        },
    )
    if result.cuda_state_numel != 0 or result.residency != "CPU_RESIDENT":
        raise RuntimeError("staged Muon initialization did not reach CPU residency")
    return result
