# SPDX-License-Identifier: Apache-2.0

"""GPU-staged AdamW for Megatron-Core's precision-aware optimizer path.

The optimizer owns FP32 master weights and Adam moments in pinned CPU slabs.
Only a bounded number of units are copied to CUDA for an update.  The model
parameters remain on CUDA and are updated before Megatron starts its normal
parameter all-gather.

This is intentionally an internal, AdamW-only implementation. Managed
asynchronous save uses a mutation fence around the authoritative CPU slabs;
asynchronous load and backward prefetch are not supported.

The instance-level factory below is fail-closed on Megatron-Core 0.17.0.  It
can be simplified once upstream exposes an optimizer builder registry and a
post-``DistributedOptimizer``-sharding ``bind_owned_params`` capability hook.
"""

from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch

from areal.engine.megatron_utils.checkpoint_snapshot import (
    DiskSnapshotBuildCleanup,
    DiskTensorRollbackSnapshot,
    SnapshotRequirement,
    preflight_snapshot_requirements,
    snapshot_parent,
    validate_snapshot_chunk_bytes,
)
from areal.engine.megatron_utils.managed_async_checkpoint import (
    ManagedAsyncSaveState,
)
from areal.engine.megatron_utils.optimizer_chain import (
    get_managed_base_optimizer as get_managed_base_optimizer,
)
from areal.engine.megatron_utils.optimizer_chain import (
    iter_megatron_optimizer_leaves,
)

_SUPPORTED_MEGATRON_CORE_VERSION = "0.17.0"


@dataclass(frozen=True)
class GPUStagedAdamWConfig:
    """Internal configuration for the GPU-staged AdamW implementation."""

    buffer_count: int = 2
    bucket_size_mb: float = 128.0
    checkpoint_snapshot_root: str | None = None
    checkpoint_snapshot_chunk_mb: float = 64.0

    def __post_init__(self) -> None:
        if self.buffer_count < 1:
            raise ValueError("buffer_count must be at least 1")
        if self.bucket_size_mb <= 0:
            raise ValueError("bucket_size_mb must be positive")
        if self.checkpoint_snapshot_chunk_mb <= 0:
            raise ValueError("checkpoint_snapshot_chunk_mb must be positive")
        validate_snapshot_chunk_bytes(self.checkpoint_snapshot_chunk_bytes)

    @property
    def bucket_numel(self) -> int:
        return max(1, int(self.bucket_size_mb * 1024 * 1024) // 4)

    @property
    def checkpoint_snapshot_chunk_bytes(self) -> int:
        raw_bytes = self.checkpoint_snapshot_chunk_mb * 1024 * 1024
        chunk_bytes = int(raw_bytes)
        if raw_bytes != chunk_bytes or chunk_bytes % 4:
            raise ValueError(
                "checkpoint snapshot chunk size must be an exact FP32-aligned "
                f"byte count, got {raw_bytes!r} bytes"
            )
        return chunk_bytes


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
    """Three flat, pinned FP32 slabs which are authoritative between steps."""

    master: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor

    @classmethod
    def allocate(cls, numel: int) -> AdamWCPUSlabs:
        if numel < 0:
            raise ValueError("numel must be non-negative")
        kwargs = {"dtype": torch.float32, "device": "cpu", "pin_memory": True}
        return cls(
            master=torch.empty(numel, **kwargs),
            exp_avg=torch.zeros(numel, **kwargs),
            exp_avg_sq=torch.zeros(numel, **kwargs),
        )


class _CheckpointLifecycle(Enum):
    CLEAN = auto()
    SNAPSHOT_ACTIVE = auto()
    LOAD_ACTIVE = auto()
    RECOVERY_PENDING = auto()
    POISONED = auto()
    RELOAD_REQUIRED = auto()
    COMMIT_DECIDED = auto()
    CLEANUP_PENDING = auto()


class _RollbackActionStatus(Enum):
    PENDING = auto()
    COMPLETED = auto()


@dataclass(frozen=True)
class _ParamGroupRollbackSnapshot:
    metadata: dict[str, Any]
    params: Any


@dataclass(frozen=True)
class _RuntimeRollbackSnapshot:
    loaded_state: dict[torch.Tensor, set[str]]
    prepared: bool
    residency: str
    lifecycle: _CheckpointLifecycle
    load_error: BaseException | None
    recovery_poisoned: bool


@dataclass
class _CheckpointRollbackAction:
    name: str
    target: Any
    snapshot: Any | None
    restore: Callable[[Any, Any], None]
    dependencies: tuple[str, ...] = ()
    status: _RollbackActionStatus = _RollbackActionStatus.PENDING
    diagnostics: list[str] = field(default_factory=list)

    @property
    def pending(self) -> bool:
        return self.status is _RollbackActionStatus.PENDING

    def attempt(self) -> BaseException | None:
        if not self.pending:
            return None
        try:
            self.restore(self.target, self.snapshot)
        except BaseException as error:
            self.diagnostics[:] = [repr(error)]
            return error
        self.status = _RollbackActionStatus.COMPLETED
        # Each action owns its unique recovery copy.  Release it immediately
        # after that exact target is known to be restored.
        self.snapshot = None
        return None


@dataclass
class _CheckpointLoadRollback:
    actions: list[_CheckpointRollbackAction]
    previous_state: _CheckpointLifecycle
    previous_error: BaseException | None

    @property
    def pending_actions(self) -> tuple[_CheckpointRollbackAction, ...]:
        return tuple(action for action in self.actions if action.pending)

    @property
    def completed_action_names(self) -> frozenset[str]:
        return frozenset(action.name for action in self.actions if not action.pending)

    def _snapshot_for(self, name: str) -> Any | None:
        for action in self.actions:
            if action.name == name:
                return action.snapshot
        return None

    # Compatibility accessors retained for checkpoint tests and diagnostics.
    @property
    def master(self) -> Any | None:
        return self._snapshot_for("slab.master")

    @property
    def exp_avg(self) -> Any | None:
        return self._snapshot_for("slab.exp_avg")

    @property
    def exp_avg_sq(self) -> Any | None:
        return self._snapshot_for("slab.exp_avg_sq")


@dataclass
class _CheckpointCleanupJournal:
    """Strong references awaiting release after the irreversible commit decision."""

    references: list[Any]


@dataclass
class _CheckpointCommitToken:
    decided: bool = False


def _restore_checkpoint_drain(target: GPUStagedAdamW, snapshot: Any) -> None:
    del snapshot
    target.drain()


def _restore_checkpoint_slab(target: tuple[GPUStagedAdamW, str], snapshot: Any) -> None:
    optimizer, slab_name = target
    assert optimizer.cpu_slabs is not None
    destination = getattr(optimizer.cpu_slabs, slab_name)
    restore = getattr(snapshot, "restore_into", None)
    if callable(restore):
        restore(destination)
    else:
        destination.copy_(snapshot)


def _restore_checkpoint_param_group(
    target: dict[str, Any], snapshot: _ParamGroupRollbackSnapshot
) -> None:
    target.clear()
    target.update(snapshot.metadata)
    target["params"] = snapshot.params


def _restore_checkpoint_runtime_metadata(
    target: GPUStagedAdamW, snapshot: _RuntimeRollbackSnapshot
) -> None:
    target._checkpoint_loaded_state = {
        param: set(fields) for param, fields in snapshot.loaded_state.items()
    }
    target._checkpoint_prepared = snapshot.prepared
    target._residency = snapshot.residency
    target._checkpoint_lifecycle = snapshot.lifecycle
    target._checkpoint_load_error = snapshot.load_error
    target._checkpoint_recovery_poisoned = snapshot.recovery_poisoned


class _SlotPhase(Enum):
    FREE = auto()
    D2H_PENDING = auto()


class SlotStateMachine:
    """Tracks the rule that a slot cannot be reused before its D2H completes."""

    def __init__(self, slot_count: int, wait_for_slot: Callable[[int], None]) -> None:
        if slot_count < 1:
            raise ValueError("slot_count must be at least 1")
        self._phases = [_SlotPhase.FREE] * slot_count
        self._wait_for_slot = wait_for_slot

    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(phase.name for phase in self._phases)

    def acquire(self, slot_index: int) -> None:
        if self._phases[slot_index] is _SlotPhase.D2H_PENDING:
            self._wait_for_slot(slot_index)
            self._phases[slot_index] = _SlotPhase.FREE

    def mark_d2h_pending(self, slot_index: int) -> None:
        if self._phases[slot_index] is not _SlotPhase.FREE:
            raise RuntimeError(f"staging slot {slot_index} is already in use")
        self._phases[slot_index] = _SlotPhase.D2H_PENDING

    def drain(self) -> None:
        for slot_index, phase in enumerate(self._phases):
            if phase is _SlotPhase.D2H_PENDING:
                self._wait_for_slot(slot_index)
                self._phases[slot_index] = _SlotPhase.FREE


@dataclass
class _CUDAStagingSlot:
    master: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    grad: torch.Tensor
    h2d_stream: torch.cuda.Stream
    compute_stream: torch.cuda.Stream
    d2h_stream: torch.cuda.Stream
    h2d_done: torch.cuda.Event
    compute_done: torch.cuda.Event
    d2h_done: torch.cuda.Event

    @classmethod
    def allocate(cls, capacity: int, device: torch.device) -> _CUDAStagingSlot:
        tensor_kwargs = {"size": (capacity,), "dtype": torch.float32, "device": device}
        return cls(
            master=torch.empty(**tensor_kwargs),
            exp_avg=torch.empty(**tensor_kwargs),
            exp_avg_sq=torch.empty(**tensor_kwargs),
            grad=torch.empty(**tensor_kwargs),
            h2d_stream=torch.cuda.Stream(device=device),
            compute_stream=torch.cuda.Stream(device=device),
            d2h_stream=torch.cuda.Stream(device=device),
            h2d_done=torch.cuda.Event(),
            compute_done=torch.cuda.Event(),
            d2h_done=torch.cuda.Event(),
        )


class GPUStagedAdamW(torch.optim.AdamW):
    """AdamW shell whose master weights and moments live in pinned CPU slabs."""

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

        # torch.optim.AdamW creates only metadata here; tensor state remains empty.
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
        self._slots: list[_CUDAStagingSlot] = []
        self._slot_machine: SlotStateMachine | None = None
        self._bound = False
        self._residency = "UNBOUND"
        self._checkpoint_rollback: _CheckpointLoadRollback | None = None
        self._checkpoint_cleanup: _CheckpointCleanupJournal | None = None
        self._checkpoint_prepared_cleanup: _CheckpointCleanupJournal | None = None
        self._checkpoint_commit_token: Any | None = None
        self._checkpoint_attempt_token: Any | None = None
        self._checkpoint_reload_generation: Any | None = None
        self._checkpoint_snapshot_attempt_token: Any | None = None
        self._checkpoint_load_error: BaseException | None = None
        self._checkpoint_cleanup_error: str | None = None
        self._checkpoint_lifecycle = _CheckpointLifecycle.CLEAN
        self._checkpoint_loaded_state: dict[torch.Tensor, set[str]] = {}
        self._checkpoint_prepared = False
        self._checkpoint_recovery_poisoned = False
        # Resolve/create only inside a voted checkpoint preflight, never as an
        # optimizer-construction side effect.
        self._checkpoint_snapshot_parent = self.staged_config.checkpoint_snapshot_root
        self._checkpoint_snapshot_identity: Any | None = None
        self._checkpoint_build_cleanup: list[Any] = []
        self._async_save_fence: dict[str, Any] | None = None
        self._async_save_last_state = ManagedAsyncSaveState.IDLE
        self._async_save_error: BaseException | None = None

    @property
    def residency(self) -> str:
        return self._residency

    @property
    def checkpoint_lifecycle(self) -> str:
        return self._effective_checkpoint_lifecycle().name

    @property
    def checkpoint_load_attempt_token(self) -> Any | None:
        if self._checkpoint_commit_is_decided():
            return None
        return self._checkpoint_attempt_token

    @property
    def checkpoint_reload_generation(self) -> Any | None:
        return self._checkpoint_reload_generation

    @property
    def checkpoint_snapshot_attempt_token(self) -> Any | None:
        return self._checkpoint_snapshot_attempt_token

    @property
    def async_save_state(self) -> str:
        fence = self._async_save_fence
        if fence is not None:
            return fence["state"].name
        return self._async_save_last_state.name

    def _capture_async_save_sources(self) -> tuple[dict[str, Any], ...]:
        assert self.cpu_slabs is not None
        sources = []
        for name, tensor in (
            ("master", self.cpu_slabs.master),
            ("exp_avg", self.cpu_slabs.exp_avg),
            ("exp_avg_sq", self.cpu_slabs.exp_avg_sq),
        ):
            sources.append(
                {
                    "name": name,
                    "tensor": tensor,
                    "data_ptr": tensor.untyped_storage().data_ptr(),
                    "storage_nbytes": tensor.untyped_storage().nbytes(),
                    "version": tensor._version,
                    "shape": tuple(tensor.shape),
                    "dtype": tensor.dtype,
                }
            )
        return tuple(sources)

    def begin_async_checkpoint_save(
        self,
        *,
        checkpoint_id: str,
        path: str,
        control_group: Any,
        wait_fn: Callable[[], None],
        leaf_identity: Any,
    ) -> None:
        """Publish a mutation fence before exposing authoritative slab views."""
        if self._async_save_fence is not None:
            raise RuntimeError(
                "only one managed asynchronous checkpoint may be outstanding"
            )
        if self._async_save_error is not None:
            raise RuntimeError(
                "managed optimizer is fail-closed after an asynchronous save failure"
            ) from self._async_save_error
        self.prepare_checkpoint_save()
        self._async_save_fence = {
            "checkpoint_id": checkpoint_id,
            "path": path,
            "control_group": control_group,
            "leaf_identity": leaf_identity,
            "sources": self._capture_async_save_sources(),
            "group_metadata": tuple(
                {key: value for key, value in group.items() if key != "params"}
                for group in self.param_groups
            ),
            "wait_fn": wait_fn,
            "request": None,
            "call_idx": None,
            "state": ManagedAsyncSaveState.SAVE_STAGING,
        }
        self._async_save_last_state = ManagedAsyncSaveState.SAVE_STAGING

    def bind_async_checkpoint_request(self, request: Any, call_idx: int) -> None:
        fence = self._async_save_fence
        if fence is None or fence["state"] is not ManagedAsyncSaveState.SAVE_STAGING:
            raise RuntimeError("managed async checkpoint fence is not staging")
        fence["request"] = request
        fence["call_idx"] = call_idx
        fence["state"] = ManagedAsyncSaveState.SAVE_IN_FLIGHT
        self._async_save_last_state = ManagedAsyncSaveState.SAVE_IN_FLIGHT

    def _validate_async_checkpoint_sources(self) -> None:
        fence = self._async_save_fence
        if fence is None:
            raise RuntimeError("managed async checkpoint fence is missing")
        assert self.cpu_slabs is not None
        current = {
            "master": self.cpu_slabs.master,
            "exp_avg": self.cpu_slabs.exp_avg,
            "exp_avg_sq": self.cpu_slabs.exp_avg_sq,
        }
        for source in fence["sources"]:
            tensor = current[source["name"]]
            if tensor is not source["tensor"]:
                raise RuntimeError(
                    f"managed async source object changed for {source['name']}"
                )
            actual = (
                tensor.untyped_storage().data_ptr(),
                tensor.untyped_storage().nbytes(),
                tensor._version,
                tuple(tensor.shape),
                tensor.dtype,
            )
            expected = (
                source["data_ptr"],
                source["storage_nbytes"],
                source["version"],
                source["shape"],
                source["dtype"],
            )
            if actual != expected:
                raise RuntimeError(
                    "managed async checkpoint source changed while fenced: "
                    f"slab={source['name']}, expected={expected}, actual={actual}"
                )
        current_groups = tuple(
            {key: value for key, value in group.items() if key != "params"}
            for group in self.param_groups
        )
        if current_groups != fence["group_metadata"]:
            raise RuntimeError(
                "managed async checkpoint param-group metadata changed while fenced"
            )

    def complete_async_checkpoint_save(self) -> None:
        self._validate_async_checkpoint_sources()
        self._async_save_last_state = ManagedAsyncSaveState.COMPLETE
        self._async_save_fence = None
        self._async_save_error = None

    def fail_async_checkpoint_save(self, error: BaseException) -> None:
        self._async_save_last_state = ManagedAsyncSaveState.FAILED
        self._async_save_error = error
        # The manager only calls this after the worker was reaped or aborted;
        # source references can then be released without permitting mutation.
        self._async_save_fence = None

    def _wait_for_async_checkpoint_mutation(self, operation: str) -> None:
        if self._async_save_error is not None:
            raise RuntimeError(
                f"{operation} is unavailable after an asynchronous checkpoint failure"
            ) from self._async_save_error
        fence = self._async_save_fence
        if fence is None:
            return
        wait_fn = fence["wait_fn"]
        wait_fn()
        if self._async_save_error is not None:
            raise RuntimeError(
                f"{operation} is unavailable after an asynchronous checkpoint failure"
            ) from self._async_save_error
        if self._async_save_fence is not None:
            raise RuntimeError(
                f"{operation} cannot proceed while the async source is still fenced"
            )

    def configure_checkpoint_snapshot(
        self,
        *,
        parent: str | None = None,
        leaf_identity: Any,
        replacement_generation: Any | None = None,
        attempt_token: Any | None = None,
    ) -> None:
        """Bind a DP-stable identity before constructing rollback files."""
        lifecycle = self._effective_checkpoint_lifecycle()
        if lifecycle is _CheckpointLifecycle.RELOAD_REQUIRED:
            self.authorize_checkpoint_replacement(replacement_generation, attempt_token)
        elif lifecycle is _CheckpointLifecycle.CLEAN:
            if replacement_generation is not None or attempt_token is not None:
                raise RuntimeError(
                    "ordinary checkpoint snapshot cannot use replacement authority"
                )
        else:
            raise RuntimeError("checkpoint snapshot configuration requires CLEAN state")
        if parent is not None:
            self._checkpoint_snapshot_parent = parent
        self._checkpoint_snapshot_identity = leaf_identity

    def authorize_checkpoint_replacement(
        self, replacement_generation: Any, attempt_token: Any
    ) -> None:
        if (
            replacement_generation is None
            or replacement_generation is not self._checkpoint_reload_generation
            or attempt_token is None
            or getattr(replacement_generation, "active_attempt", None)
            is not attempt_token
        ):
            raise RuntimeError(
                "RELOAD_REQUIRED snapshot configuration requires matching manager "
                "replacement authority"
            )
        if (
            self._effective_checkpoint_lifecycle()
            is not _CheckpointLifecycle.RELOAD_REQUIRED
        ):
            raise RuntimeError("replacement authority requires RELOAD_REQUIRED state")
        if (
            self._checkpoint_snapshot_attempt_token is not None
            and self._checkpoint_snapshot_attempt_token is not attempt_token
        ):
            raise RuntimeError(
                "checkpoint snapshot belongs to another replacement attempt"
            )
        self._checkpoint_snapshot_attempt_token = attempt_token

    def cancel_checkpoint_snapshot_configuration(
        self, replacement_generation: Any, attempt_token: Any
    ) -> None:
        if replacement_generation is not self._checkpoint_reload_generation:
            raise RuntimeError("checkpoint replacement generation mismatch")
        if getattr(replacement_generation, "active_attempt", None) is not attempt_token:
            raise RuntimeError("checkpoint replacement attempt is no longer active")
        if self._checkpoint_snapshot_attempt_token is None:
            return
        if self._checkpoint_snapshot_attempt_token is not attempt_token:
            raise RuntimeError("checkpoint replacement attempt mismatch")
        if self._checkpoint_rollback is not None:
            raise RuntimeError("cannot cancel snapshot configuration after begin")
        self.retry_checkpoint_snapshot_build_cleanup()
        self._checkpoint_snapshot_attempt_token = None

    def _rollback_leaf_identity(self) -> Any:
        if self._checkpoint_snapshot_identity is not None:
            return self._checkpoint_snapshot_identity
        return {
            "version": 1,
            "kind": "local-layout",
            "numel": 0 if self.cpu_slabs is None else self.cpu_slabs.master.numel(),
            "groups": [
                {
                    "group_index": group_index,
                    "param_numel": [param.numel() for param in group["params"]],
                }
                for group_index, group in enumerate(self.param_groups)
            ],
        }

    def checkpoint_snapshot_requirement(self) -> SnapshotRequirement:
        if not self._bound or self.cpu_slabs is None:
            raise RuntimeError(
                "GPU-staged AdamW must be bound before snapshot preflight"
            )
        chunk_bytes = self.staged_config.checkpoint_snapshot_chunk_bytes
        required = sum(
            DiskTensorRollbackSnapshot.required_bytes(slab, chunk_bytes)
            for slab in (
                self.cpu_slabs.master,
                self.cpu_slabs.exp_avg,
                self.cpu_slabs.exp_avg_sq,
            )
        )
        return SnapshotRequirement(
            snapshot_parent(self._checkpoint_snapshot_parent), required
        )

    def preflight_checkpoint_snapshot(self) -> None:
        self.retry_checkpoint_snapshot_build_cleanup()
        preflight_snapshot_requirements((self.checkpoint_snapshot_requirement(),))

    def retry_checkpoint_snapshot_build_cleanup(self) -> None:
        """Retry files from an activation that never gained rollback authority."""
        cleanup_errors: list[BaseException] = []
        still_pending: list[Any] = []
        for cleanup in self._checkpoint_build_cleanup:
            try:
                cleanup.cleanup()
            except BaseException as error:
                cleanup_errors.append(error)
                still_pending.append(cleanup)
        self._checkpoint_build_cleanup[:] = still_pending
        if cleanup_errors:
            primary = cleanup_errors[0]
            for error in cleanup_errors[1:]:
                primary.add_note(
                    f"additional rollback snapshot build cleanup failure: {error!r}"
                )
            raise primary

    def _create_checkpoint_slab_snapshots(
        self,
    ) -> dict[str, DiskTensorRollbackSnapshot]:
        assert self.cpu_slabs is not None
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        snapshots: dict[str, DiskTensorRollbackSnapshot] = {}
        try:
            for slab_key, slab in (
                ("master", self.cpu_slabs.master),
                ("exp_avg", self.cpu_slabs.exp_avg),
                ("exp_avg_sq", self.cpu_slabs.exp_avg_sq),
            ):
                snapshots[slab_key] = DiskTensorRollbackSnapshot.create(
                    slab,
                    parent=self._checkpoint_snapshot_parent,
                    leaf_identity=self._rollback_leaf_identity(),
                    slab_key=slab_key,
                    chunk_bytes=self.staged_config.checkpoint_snapshot_chunk_bytes,
                    rank=rank,
                )
        except BaseException as original:
            partial_cleanup = getattr(original, "_areal_snapshot_build_cleanup", None)
            if isinstance(partial_cleanup, DiskSnapshotBuildCleanup):
                self._checkpoint_build_cleanup.append(partial_cleanup)
            for snapshot in reversed(tuple(snapshots.values())):
                try:
                    snapshot.cleanup()
                except BaseException as cleanup_error:
                    self._checkpoint_build_cleanup.append(snapshot.cleanup_artifact())
                    original.add_note(
                        "rollback snapshot partial-create cleanup failed: "
                        f"{cleanup_error!r}"
                    )
            raise
        return snapshots

    def _install_checkpoint_rollback(
        self,
        slab_snapshots: Mapping[str, DiskTensorRollbackSnapshot],
        previous_lifecycle: _CheckpointLifecycle,
    ) -> None:
        runtime_snapshot = _RuntimeRollbackSnapshot(
            loaded_state={
                param: set(fields)
                for param, fields in self._checkpoint_loaded_state.items()
            },
            prepared=self._checkpoint_prepared,
            residency=self._residency,
            lifecycle=previous_lifecycle,
            load_error=self._checkpoint_load_error,
            recovery_poisoned=self._checkpoint_recovery_poisoned,
        )
        actions = [
            _CheckpointRollbackAction(
                "runtime.drain", self, self._residency, _restore_checkpoint_drain
            ),
            _CheckpointRollbackAction(
                "slab.master",
                (self, "master"),
                slab_snapshots["master"],
                _restore_checkpoint_slab,
                dependencies=("runtime.drain",),
            ),
            _CheckpointRollbackAction(
                "slab.exp_avg",
                (self, "exp_avg"),
                slab_snapshots["exp_avg"],
                _restore_checkpoint_slab,
                dependencies=("runtime.drain",),
            ),
            _CheckpointRollbackAction(
                "slab.exp_avg_sq",
                (self, "exp_avg_sq"),
                slab_snapshots["exp_avg_sq"],
                _restore_checkpoint_slab,
                dependencies=("runtime.drain",),
            ),
        ]
        actions.extend(
            _CheckpointRollbackAction(
                f"param_group.{group_index}",
                group,
                _ParamGroupRollbackSnapshot(
                    metadata={
                        key: value for key, value in group.items() if key != "params"
                    },
                    params=group["params"],
                ),
                _restore_checkpoint_param_group,
            )
            for group_index, group in enumerate(self.param_groups)
        )
        metadata_dependencies = tuple(action.name for action in actions)
        actions.append(
            _CheckpointRollbackAction(
                "runtime.metadata",
                self,
                runtime_snapshot,
                _restore_checkpoint_runtime_metadata,
                dependencies=metadata_dependencies,
            )
        )
        rollback = _CheckpointLoadRollback(
            actions=actions,
            previous_state=previous_lifecycle,
            previous_error=self._checkpoint_load_error,
        )
        loaded_state = {layout.param: set() for layout in self._layouts}
        self._checkpoint_rollback = rollback
        self._checkpoint_loaded_state = loaded_state
        self._checkpoint_prepared = False
        self._checkpoint_lifecycle = _CheckpointLifecycle.LOAD_ACTIVE

    def _effective_checkpoint_lifecycle(self) -> _CheckpointLifecycle:
        """Project the shared commit decision onto this leaf immediately."""
        if self._checkpoint_commit_is_decided():
            if self._checkpoint_lifecycle not in (
                _CheckpointLifecycle.COMMIT_DECIDED,
                _CheckpointLifecycle.CLEANUP_PENDING,
                _CheckpointLifecycle.CLEAN,
            ):
                self._checkpoint_lifecycle = _CheckpointLifecycle.COMMIT_DECIDED
            # Rollback is permanently forbidden, so a pre-commit failure can
            # no longer affect trainability or retain an exception traceback.
            self._checkpoint_load_error = None
            self._checkpoint_recovery_poisoned = False
        return self._checkpoint_lifecycle

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
        capacity = max((unit.numel for unit in self._units), default=1)
        device = next(iter(devices), empty_device)
        assert device is not None
        if device.type != "cuda":
            raise ValueError("staged AdamW device must be CUDA")
        if self._units:
            self._slots = [
                _CUDAStagingSlot.allocate(capacity, device)
                for _ in range(self.staged_config.buffer_count)
            ]
            self._slot_machine = SlotStateMachine(len(self._slots), self._wait_for_slot)

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
        self._residency = "STEP_ACTIVE"
        if self._units:
            caller_stream = torch.cuda.current_stream(device)
            params_ready = torch.cuda.Event()
            params_ready.record(caller_stream)
            for unit_index, unit in enumerate(self._units):
                self._schedule_master_initialization(
                    unit, unit_index % len(self._slots), params_ready
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

    def _wait_for_slot(self, slot_index: int) -> None:
        self._slots[slot_index].d2h_done.synchronize()

    def _schedule_master_initialization(
        self,
        unit: _UpdateUnit,
        slot_index: int,
        params_ready: torch.cuda.Event,
    ) -> None:
        assert self.cpu_slabs is not None and self._slot_machine is not None
        self._slot_machine.acquire(slot_index)
        slot = self._slots[slot_index]
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
        self._slot_machine.mark_d2h_pending(slot_index)

    def _schedule_update(
        self,
        unit: _UpdateUnit,
        slot_index: int,
        grads_ready: torch.cuda.Event,
    ) -> None:
        assert self.cpu_slabs is not None and self._slot_machine is not None
        self._slot_machine.acquire(slot_index)
        slot = self._slots[slot_index]
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

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            for part in active_parts:
                master_part = slot.master.narrow(0, part.unit_offset, part.numel)
                exp_avg_part = slot.exp_avg.narrow(0, part.unit_offset, part.numel)
                exp_avg_sq_part = slot.exp_avg_sq.narrow(
                    0, part.unit_offset, part.numel
                )
                grad_part = slot.grad.narrow(0, part.unit_offset, part.numel)
                if weight_decay != 0.0:
                    master_part.mul_(1.0 - lr * weight_decay)
                exp_avg_part.mul_(beta1).add_(grad_part, alpha=1.0 - beta1)
                exp_avg_sq_part.mul_(beta2).addcmul_(
                    grad_part, grad_part, value=1.0 - beta2
                )
                denom = exp_avg_sq_part.sqrt().div_(bias_correction2**0.5).add_(eps)
                master_part.addcdiv_(exp_avg_part, denom, value=-lr / bias_correction1)
                part.param.detach().view(-1).narrow(
                    0, part.param_offset, part.numel
                ).copy_(slot.master.narrow(0, part.unit_offset, part.numel))
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
        self._slot_machine.mark_d2h_pending(slot_index)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        self._wait_for_async_checkpoint_mutation("optimizer step")
        if not self._bound:
            raise RuntimeError("bind_owned_params() must be called before step()")
        checkpoint_lifecycle = self._effective_checkpoint_lifecycle()
        if checkpoint_lifecycle in (
            _CheckpointLifecycle.SNAPSHOT_ACTIVE,
            _CheckpointLifecycle.RECOVERY_PENDING,
            _CheckpointLifecycle.POISONED,
            _CheckpointLifecycle.RELOAD_REQUIRED,
        ):
            raise RuntimeError(
                "GPU-staged AdamW is unavailable after a failed checkpoint load"
            ) from self._checkpoint_load_error
        if checkpoint_lifecycle is _CheckpointLifecycle.LOAD_ACTIVE:
            raise RuntimeError("checkpoint load transaction is still active")
        loss = closure() if closure is not None else None
        if not self._units:
            self._residency = "CPU_RESIDENT"
            return loss
        for group in self.param_groups:
            if group["params"]:
                group["step"] = int(group.get("step", 0)) + 1
        device = self._slots[0].master.device
        caller_stream = torch.cuda.current_stream(device)
        grads_ready = torch.cuda.Event()
        grads_ready.record(caller_stream)
        self._residency = "STEP_ACTIVE"
        for unit_index, unit in enumerate(self._units):
            self._schedule_update(unit, unit_index % len(self._slots), grads_ready)

        # Megatron starts parameter all-gather immediately after inner step.
        # Queue model-shard visibility on the exact calling stream so the
        # following parameter all-gather cannot race. State D2H remains async.
        for slot_index, phase in enumerate(self._slot_machine.phases):
            if phase == _SlotPhase.D2H_PENDING.name:
                caller_stream.wait_event(self._slots[slot_index].compute_done)
        return loss

    def drain(self) -> None:
        if self._slot_machine is not None:
            self._slot_machine.drain()
        if self._bound:
            self._residency = "CPU_RESIDENT"

    def offload_to_cpu(self) -> None:
        self.drain()

    def restore_from_cpu(self) -> None:
        return

    def prepare_checkpoint_save(self) -> None:
        """Fence D2H and validate the synchronous CPU checkpoint source."""
        self._wait_for_async_checkpoint_mutation("checkpoint save")
        if not self._bound or self.cpu_slabs is None:
            raise RuntimeError("GPU-staged AdamW must be bound before checkpoint save")
        self.retry_checkpoint_snapshot_build_cleanup()
        checkpoint_lifecycle = self._effective_checkpoint_lifecycle()
        if checkpoint_lifecycle is _CheckpointLifecycle.LOAD_ACTIVE:
            raise RuntimeError("cannot save during a checkpoint load transaction")
        if checkpoint_lifecycle is _CheckpointLifecycle.SNAPSHOT_ACTIVE:
            raise RuntimeError("cannot save while rollback snapshot is being created")
        if checkpoint_lifecycle in (
            _CheckpointLifecycle.COMMIT_DECIDED,
            _CheckpointLifecycle.CLEANUP_PENDING,
        ):
            raise RuntimeError(
                "checkpoint snapshot cleanup must finish before another save"
            )
        if checkpoint_lifecycle in (
            _CheckpointLifecycle.RECOVERY_PENDING,
            _CheckpointLifecycle.POISONED,
            _CheckpointLifecycle.RELOAD_REQUIRED,
        ):
            raise RuntimeError("cannot save after a failed checkpoint load") from (
                self._checkpoint_load_error
            )
        self.drain()
        if self.cuda_state_numel != 0:
            raise RuntimeError(
                "managed optimizer checkpoint source contains CUDA state"
            )
        self._validate_bound_state_views()

    def begin_checkpoint_load(
        self,
        *,
        attempt_token: Any | None = None,
        replacement_generation: Any | None = None,
    ) -> None:
        """Snapshot CPU state so a failed DCP load cannot remain partially usable."""
        self._wait_for_async_checkpoint_mutation("checkpoint load")
        if not self._bound or self.cpu_slabs is None:
            raise RuntimeError("GPU-staged AdamW must be bound before checkpoint load")
        checkpoint_lifecycle = self._effective_checkpoint_lifecycle()
        if checkpoint_lifecycle is _CheckpointLifecycle.RECOVERY_PENDING:
            raise RuntimeError(
                "retained checkpoint rollback must be recovered before loading"
            ) from self._checkpoint_load_error
        if checkpoint_lifecycle is _CheckpointLifecycle.POISONED:
            raise RuntimeError(
                "poisoned optimizer requires an explicit full checkpoint recovery"
            ) from self._checkpoint_load_error
        if checkpoint_lifecycle is _CheckpointLifecycle.LOAD_ACTIVE:
            raise RuntimeError("checkpoint load transaction is already active")
        if checkpoint_lifecycle is _CheckpointLifecycle.SNAPSHOT_ACTIVE:
            raise RuntimeError("checkpoint rollback snapshot is already being created")
        if checkpoint_lifecycle in (
            _CheckpointLifecycle.COMMIT_DECIDED,
            _CheckpointLifecycle.CLEANUP_PENDING,
        ):
            raise RuntimeError(
                "checkpoint snapshot cleanup must finish before another load"
            )
        if self._checkpoint_attempt_token is not None:
            raise RuntimeError(
                "checkpoint load already belongs to another begin attempt"
            )
        if checkpoint_lifecycle is _CheckpointLifecycle.RELOAD_REQUIRED:
            if (
                replacement_generation is None
                or replacement_generation is not self._checkpoint_reload_generation
                or attempt_token is None
                or getattr(replacement_generation, "active_attempt", None)
                is not attempt_token
                or self._checkpoint_snapshot_attempt_token is not attempt_token
            ):
                raise RuntimeError(
                    "RELOAD_REQUIRED load requires matching configured replacement "
                    "generation and attempt"
                )
        elif (
            replacement_generation is not None
            or self._checkpoint_snapshot_attempt_token is not None
        ):
            raise RuntimeError("ordinary load cannot use replacement authority")
        self.preflight_checkpoint_snapshot()
        previous_lifecycle = checkpoint_lifecycle
        self._checkpoint_attempt_token = (
            attempt_token if attempt_token is not None else object()
        )
        self._checkpoint_lifecycle = _CheckpointLifecycle.SNAPSHOT_ACTIVE
        slab_snapshots: dict[str, DiskTensorRollbackSnapshot] = {}
        try:
            self.drain()
            # Publish rollback authority only after all three sealed files have
            # passed a full read-back checksum verification.
            slab_snapshots = self._create_checkpoint_slab_snapshots()
            self._install_checkpoint_rollback(
                slab_snapshots, previous_lifecycle=previous_lifecycle
            )
        except BaseException as original:
            for snapshot in reversed(tuple(slab_snapshots.values())):
                try:
                    snapshot.cleanup()
                except BaseException as cleanup_error:
                    self._checkpoint_build_cleanup.append(snapshot.cleanup_artifact())
                    original.add_note(
                        "rollback snapshot activation cleanup failed: "
                        f"{cleanup_error!r}"
                    )
            self._checkpoint_lifecycle = previous_lifecycle
            self._checkpoint_attempt_token = None
            self._checkpoint_snapshot_attempt_token = None
            raise

    def prepare_checkpoint_load(self) -> None:
        """Validate a load while retaining the rollback snapshot."""
        if self._checkpoint_commit_is_decided():
            raise RuntimeError(
                "checkpoint commit is decided; only snapshot cleanup may continue"
            )
        if self._checkpoint_rollback is None:
            raise RuntimeError("no checkpoint load transaction is active")
        self.drain()
        assert self.cpu_slabs is not None
        for name, slab in (
            ("master", self.cpu_slabs.master),
            ("exp_avg", self.cpu_slabs.exp_avg),
            ("exp_avg_sq", self.cpu_slabs.exp_avg_sq),
        ):
            if slab.device.type != "cpu" or slab.dtype is not torch.float32:
                raise RuntimeError(f"loaded {name} slab must remain CPU FP32")
            if not slab.is_pinned():
                raise RuntimeError(f"loaded {name} slab lost pinned residency")
        if self.cuda_state_numel != 0:
            raise RuntimeError("checkpoint load created CUDA optimizer state")
        expected = {"master_param", "exp_avg", "exp_avg_sq"}
        missing = {
            index: sorted(expected - self._checkpoint_loaded_state[layout.param])
            for index, layout in enumerate(self._layouts)
            if self._checkpoint_loaded_state[layout.param] != expected
        }
        if missing:
            raise ValueError(
                f"managed optimizer checkpoint is missing parameter state: {missing}"
            )
        self._validate_bound_state_views()
        self._checkpoint_prepared = True

    def prepare_checkpoint_commit(self, commit_token: Any | None = None) -> None:
        """Validate commit readiness without releasing rollback state."""
        if self._checkpoint_commit_is_decided():
            raise RuntimeError(
                "checkpoint commit is already decided; rollback token is immutable"
            )
        if self._checkpoint_rollback is None or not self._checkpoint_prepared:
            raise RuntimeError("checkpoint load was not prepared for commit")
        if self._checkpoint_commit_token is not None:
            if (
                commit_token is not None
                and commit_token is not self._checkpoint_commit_token
            ):
                raise RuntimeError("checkpoint commit token cannot be replaced")
            if self._checkpoint_prepared_cleanup is not None:
                return
        references = []
        for action in self._checkpoint_rollback.actions:
            snapshot = action.snapshot
            if snapshot is None:
                continue
            cleanup_artifact = getattr(snapshot, "cleanup_artifact", None)
            references.append(
                cleanup_artifact() if callable(cleanup_artifact) else snapshot
            )
        self._checkpoint_prepared_cleanup = _CheckpointCleanupJournal(references)
        if self._checkpoint_commit_token is None:
            self._checkpoint_commit_token = commit_token or _CheckpointCommitToken()

    def _checkpoint_commit_is_decided(self) -> bool:
        return bool(
            self._checkpoint_commit_token is not None
            and getattr(self._checkpoint_commit_token, "decided", False)
        )

    def decide_checkpoint_commit(self) -> None:
        """Materialize cleanup bookkeeping for an already committed leaf."""
        if (
            self._checkpoint_lifecycle
            in (
                _CheckpointLifecycle.COMMIT_DECIDED,
                _CheckpointLifecycle.CLEANUP_PENDING,
                _CheckpointLifecycle.CLEAN,
            )
            and self._checkpoint_rollback is None
        ):
            return
        if not self._checkpoint_commit_is_decided():
            raise RuntimeError("checkpoint commit decision has not been published")
        assert self._checkpoint_rollback is not None
        if self._checkpoint_prepared_cleanup is None:
            raise RuntimeError("checkpoint cleanup journal was not prepared")
        self._checkpoint_lifecycle = _CheckpointLifecycle.COMMIT_DECIDED
        if self._checkpoint_cleanup is None:
            self._checkpoint_cleanup = self._checkpoint_prepared_cleanup
        # From this assignment onward no API retains restore targets or
        # callables. The old state can only be released, never replayed.
        self._checkpoint_rollback = None
        self._checkpoint_prepared_cleanup = None
        self._checkpoint_attempt_token = None
        self._checkpoint_load_error = None
        self._checkpoint_cleanup_error = None
        self._checkpoint_loaded_state.clear()
        self._checkpoint_prepared = False
        self._checkpoint_recovery_poisoned = False
        self._residency = "CPU_RESIDENT"
        self._checkpoint_lifecycle = _CheckpointLifecycle.CLEANUP_PENDING

    def record_checkpoint_cleanup_error(self, error: BaseException) -> None:
        """Keep one traceback-free diagnostic for the latest cleanup failure."""
        if not self._checkpoint_commit_is_decided():
            raise RuntimeError("cleanup diagnostics require a committed checkpoint")
        self._checkpoint_cleanup_error = f"{type(error).__name__}: {error}"

    def discard_checkpoint_snapshot(self) -> None:
        """Idempotently release cleanup-only references after commit."""
        if self._checkpoint_lifecycle is _CheckpointLifecycle.CLEAN:
            if self._checkpoint_cleanup is not None:
                raise RuntimeError("clean optimizer retained a cleanup journal")
            return
        if self._checkpoint_lifecycle not in (
            _CheckpointLifecycle.COMMIT_DECIDED,
            _CheckpointLifecycle.CLEANUP_PENDING,
        ):
            raise RuntimeError(
                "checkpoint snapshot cleanup requires a committed transaction"
            )
        if self._checkpoint_cleanup is None:
            raise RuntimeError("committed optimizer is missing its cleanup journal")
        self._checkpoint_lifecycle = _CheckpointLifecycle.CLEANUP_PENDING
        cleanup_errors: list[BaseException] = []
        pending_references: list[Any] = []
        for reference in self._checkpoint_cleanup.references:
            cleanup = getattr(reference, "cleanup", None)
            if not callable(cleanup):
                continue
            try:
                cleanup()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                pending_references.append(reference)
        self._checkpoint_cleanup.references[:] = pending_references
        if cleanup_errors:
            primary = cleanup_errors[0]
            for cleanup_error in cleanup_errors[1:]:
                primary.add_note(
                    f"additional rollback snapshot cleanup failure: {cleanup_error!r}"
                )
            self.record_checkpoint_cleanup_error(primary)
            raise primary
        self._checkpoint_cleanup = None
        self._checkpoint_prepared_cleanup = None
        self._checkpoint_rollback = None
        self._checkpoint_commit_token = None
        self._checkpoint_load_error = None
        self._checkpoint_cleanup_error = None
        self._checkpoint_loaded_state.clear()
        self._checkpoint_prepared = False
        self._checkpoint_recovery_poisoned = False
        self._checkpoint_lifecycle = _CheckpointLifecycle.CLEAN
        self._checkpoint_reload_generation = None
        self._checkpoint_snapshot_attempt_token = None

    def commit_checkpoint_load(self) -> None:
        """Compatibility helper for callers without a global coordinator."""
        if self._checkpoint_commit_is_decided():
            self.decide_checkpoint_commit()
            self.discard_checkpoint_snapshot()
            return
        commit_token = _CheckpointCommitToken()
        self.prepare_checkpoint_commit(commit_token)
        commit_token.decided = True
        self.decide_checkpoint_commit()
        self.discard_checkpoint_snapshot()

    def complete_checkpoint_load(self) -> None:
        """Backward-compatible single-leaf prepare and commit."""
        if self._checkpoint_commit_is_decided():
            self.commit_checkpoint_load()
            return
        self.prepare_checkpoint_load()
        self.commit_checkpoint_load()

    def abort_checkpoint_load(
        self,
        error: BaseException,
        *,
        poison: bool = False,
        attempt_token: Any | None = None,
        replacement_generation: Any | None = None,
    ) -> None:
        """Best-effort restore of the pre-load CPU snapshot."""
        if (
            self._checkpoint_commit_is_decided()
            or self._checkpoint_lifecycle
            in (
                _CheckpointLifecycle.COMMIT_DECIDED,
                _CheckpointLifecycle.CLEANUP_PENDING,
            )
            or self._checkpoint_cleanup is not None
        ):
            raise RuntimeError(
                "cannot abort after the irreversible checkpoint commit decision"
            )
        if (
            attempt_token is not None
            and self._checkpoint_attempt_token is not None
            and attempt_token is not self._checkpoint_attempt_token
        ):
            raise RuntimeError("checkpoint abort belongs to a different begin attempt")
        if self._checkpoint_reload_generation is not None:
            if replacement_generation is not self._checkpoint_reload_generation:
                raise RuntimeError("checkpoint abort replacement generation mismatch")
        elif replacement_generation is not None:
            raise RuntimeError(
                "ordinary checkpoint abort received replacement authority"
            )
        if attempt_token is not None and self._checkpoint_attempt_token is None:
            return
        rollback = self._checkpoint_rollback
        if rollback is None:
            if self._checkpoint_attempt_token is not None:
                self.retry_checkpoint_snapshot_build_cleanup()
                self._checkpoint_attempt_token = None
                self._checkpoint_snapshot_attempt_token = None
            if poison:
                self.mark_checkpoint_poisoned(error)
            return
        if self._checkpoint_attempt_token is None:
            raise RuntimeError("checkpoint rollback is missing its begin authority")
        self._checkpoint_prepared_cleanup = None
        self._checkpoint_commit_token = None
        rollback_errors: list[tuple[_CheckpointRollbackAction, BaseException]] = []
        for action in rollback.actions:
            if not set(action.dependencies).issubset(rollback.completed_action_names):
                continue
            rollback_error = action.attempt()
            if rollback_error is None:
                continue
            rollback_errors.append((action, rollback_error))
            error.add_note(
                "GPU-staged optimizer rollback action "
                f"{action.name!r} failed: {rollback_error!r}"
            )
        if rollback_errors:
            self._checkpoint_lifecycle = _CheckpointLifecycle.RECOVERY_PENDING
            self._checkpoint_load_error = error
            self._checkpoint_recovery_poisoned = True
            self._checkpoint_prepared = False
            self._residency = "CPU_RESIDENT"
            raise rollback_errors[0][1]
        if rollback.pending_actions:
            raise RuntimeError("checkpoint rollback retained unattempted actions")
        self._checkpoint_rollback = None
        self._checkpoint_attempt_token = None
        self._checkpoint_snapshot_attempt_token = None
        if rollback.previous_state is _CheckpointLifecycle.RELOAD_REQUIRED and not (
            poison or self._checkpoint_recovery_poisoned
        ):
            self._checkpoint_lifecycle = _CheckpointLifecycle.RELOAD_REQUIRED
            self._checkpoint_load_error = rollback.previous_error
        elif (
            poison
            or self._checkpoint_recovery_poisoned
            or rollback.previous_state is _CheckpointLifecycle.POISONED
        ):
            self._checkpoint_lifecycle = _CheckpointLifecycle.POISONED
            self._checkpoint_load_error = error
        else:
            self._checkpoint_lifecycle = _CheckpointLifecycle.CLEAN
            self._checkpoint_load_error = rollback.previous_error
        self._checkpoint_recovery_poisoned = False

    def retry_checkpoint_recovery(self, *, attempt_token: Any | None = None) -> None:
        """Retry a retained CPU snapshot restore before a new full load."""
        if self._checkpoint_commit_is_decided() or self._checkpoint_cleanup is not None:
            raise RuntimeError("committed checkpoint cleanup is not rollback recovery")
        if self._checkpoint_rollback is None:
            if self._checkpoint_attempt_token is not None:
                if (
                    attempt_token is not None
                    and attempt_token is not self._checkpoint_attempt_token
                ):
                    raise RuntimeError(
                        "checkpoint recovery belongs to a different begin attempt"
                    )
                self.retry_checkpoint_snapshot_build_cleanup()
                self._checkpoint_attempt_token = None
            return
        assert self._checkpoint_load_error is not None
        self.abort_checkpoint_load(
            self._checkpoint_load_error,
            poison=True,
            attempt_token=attempt_token,
            replacement_generation=self._checkpoint_reload_generation,
        )

    def prepare_checkpoint_recovery(
        self,
        *,
        attempt_token: Any | None = None,
        reload_generation: Any | None = None,
    ) -> None:
        """Recover retained snapshots, then require a unanimous full reload."""
        if (
            self._checkpoint_commit_is_decided()
            or self._checkpoint_lifecycle
            in (
                _CheckpointLifecycle.COMMIT_DECIDED,
                _CheckpointLifecycle.CLEANUP_PENDING,
            )
            or self._checkpoint_cleanup is not None
        ):
            raise RuntimeError(
                "committed checkpoint cleanup cannot enter rollback recovery"
            )
        if self._checkpoint_lifecycle is _CheckpointLifecycle.LOAD_ACTIVE:
            raise RuntimeError("cannot recover an unclassified active checkpoint load")
        if (
            self._checkpoint_rollback is not None
            or self._checkpoint_attempt_token is not None
        ):
            self.retry_checkpoint_recovery(attempt_token=attempt_token)
        if self._checkpoint_rollback is not None:
            raise RuntimeError("checkpoint rollback snapshot recovery is incomplete")
        if self._checkpoint_lifecycle in (
            _CheckpointLifecycle.CLEAN,
            _CheckpointLifecycle.POISONED,
            _CheckpointLifecycle.RELOAD_REQUIRED,
        ):
            if reload_generation is None:
                if self._checkpoint_reload_generation is None:
                    from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
                        ManagedCheckpointReloadGeneration,
                    )

                    reload_generation = ManagedCheckpointReloadGeneration()
                else:
                    reload_generation = self._checkpoint_reload_generation
            if (
                self._checkpoint_reload_generation is not None
                and self._checkpoint_reload_generation is not reload_generation
            ):
                raise RuntimeError("checkpoint recovery generation mismatch")
            self._checkpoint_reload_generation = reload_generation
            self._checkpoint_snapshot_attempt_token = None
            self._checkpoint_lifecycle = _CheckpointLifecycle.RELOAD_REQUIRED
            self._residency = "CPU_RESIDENT"
            return
        raise RuntimeError(
            f"cannot enter checkpoint recovery from {self.checkpoint_lifecycle}"
        )

    def mark_checkpoint_poisoned(self, error: BaseException) -> None:
        if (
            self._checkpoint_commit_is_decided()
            or self._checkpoint_lifecycle
            in (
                _CheckpointLifecycle.COMMIT_DECIDED,
                _CheckpointLifecycle.CLEANUP_PENDING,
            )
            or self._checkpoint_cleanup is not None
        ):
            # A control-plane failure after commit may disable future
            # checkpoints, but it cannot restore or invalidate the new state.
            self._checkpoint_cleanup_error = f"{type(error).__name__}: {error}"
            return
        if self._checkpoint_rollback is not None:
            self._checkpoint_lifecycle = _CheckpointLifecycle.POISONED
            self._checkpoint_recovery_poisoned = True
            self._checkpoint_load_error = error
            self._residency = "CPU_RESIDENT"
            return
        self._checkpoint_lifecycle = _CheckpointLifecycle.POISONED
        self._checkpoint_load_error = error
        self._residency = "CPU_RESIDENT"

    def apply_model_checkpoint_reset(self) -> None:
        """Stream loaded model shards while retaining all pre-load state."""
        self._wait_for_async_checkpoint_mutation("model-only optimizer reset")
        if not self._bound or self.cpu_slabs is None or self._slot_machine is None:
            raise RuntimeError("GPU-staged AdamW must be bound before state reset")
        checkpoint_lifecycle = self._effective_checkpoint_lifecycle()
        if checkpoint_lifecycle in (
            _CheckpointLifecycle.COMMIT_DECIDED,
            _CheckpointLifecycle.CLEANUP_PENDING,
        ):
            raise RuntimeError(
                "model-only reset is unavailable while checkpoint cleanup is pending"
            )
        if checkpoint_lifecycle is not _CheckpointLifecycle.LOAD_ACTIVE:
            raise RuntimeError("model-only reset requires an active load transaction")
        self.drain()
        device = self._slots[0].master.device
        caller_stream = torch.cuda.current_stream(device)
        params_ready = torch.cuda.Event()
        params_ready.record(caller_stream)
        self._residency = "STEP_ACTIVE"
        for unit_index, unit in enumerate(self._units):
            self._schedule_master_initialization(
                unit, unit_index % len(self._slots), params_ready
            )
        self.drain()

    def finalize_model_checkpoint_reset(self) -> None:
        """Reset moments and step only after every leaf rebuilt its master."""
        self._wait_for_async_checkpoint_mutation("model-only optimizer reset")
        checkpoint_lifecycle = self._effective_checkpoint_lifecycle()
        if checkpoint_lifecycle in (
            _CheckpointLifecycle.COMMIT_DECIDED,
            _CheckpointLifecycle.CLEANUP_PENDING,
        ):
            raise RuntimeError(
                "model-only reset is unavailable while checkpoint cleanup is pending"
            )
        if checkpoint_lifecycle is not _CheckpointLifecycle.LOAD_ACTIVE:
            raise RuntimeError("model-only reset requires an active load transaction")
        assert self.cpu_slabs is not None
        self.cpu_slabs.exp_avg.zero_()
        self.cpu_slabs.exp_avg_sq.zero_()
        for group in self.param_groups:
            group["step"] = 0
        expected = {"master_param", "exp_avg", "exp_avg_sq"}
        self._checkpoint_loaded_state = {
            layout.param: set(expected) for layout in self._layouts
        }

    def reset_from_model_params(self) -> None:
        """Backward-compatible transactional single-leaf model reset."""
        self.begin_checkpoint_load()
        try:
            self.apply_model_checkpoint_reset()
            self.finalize_model_checkpoint_reset()
            self.prepare_checkpoint_load()
            self.commit_checkpoint_load()
        except BaseException as error:
            try:
                self.abort_checkpoint_load(error, poison=True)
            except BaseException:
                pass
            raise

    def get_unscaled_state(self, param: torch.Tensor, key: str) -> torch.Tensor:
        fence = self._async_save_fence
        if fence is None or fence["state"] is not ManagedAsyncSaveState.SAVE_STAGING:
            self._wait_for_async_checkpoint_mutation("mutable optimizer state access")
        self.drain()
        return self.state[param][key]

    def set_scaled_state(
        self, param: torch.Tensor, key: str, value: torch.Tensor
    ) -> None:
        self._wait_for_async_checkpoint_mutation("set_scaled_state")
        self._ensure_checkpoint_state_mutation_allowed("set_scaled_state")
        self.drain()
        if key not in ("master_param", "exp_avg", "exp_avg_sq"):
            raise KeyError(f"unsupported staged AdamW checkpoint state: {key}")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"loaded {key} must be a tensor")
        if value.device.type != "cpu":
            raise RuntimeError("managed optimizer checkpoint state must load from CPU")
        if value.dtype is not torch.float32:
            raise TypeError(f"loaded {key} dtype must be torch.float32")
        destination = self.state[param][key]
        if destination.shape != value.shape or destination.numel() != value.numel():
            raise ValueError(
                f"checkpoint state shape mismatch for {key}: "
                f"expected {tuple(destination.shape)}, got {tuple(value.shape)}"
            )
        destination.copy_(value)
        if self._checkpoint_lifecycle is _CheckpointLifecycle.LOAD_ACTIVE:
            if param not in self._checkpoint_loaded_state:
                raise KeyError("checkpoint parameter is not owned by this optimizer")
            self._checkpoint_loaded_state[param].add(key)

    def state_dict(self) -> dict[str, Any]:
        if self._bound:
            if (
                self._effective_checkpoint_lifecycle()
                is _CheckpointLifecycle.LOAD_ACTIVE
            ):
                self.drain()
            elif (
                self._async_save_fence is not None
                and self._async_save_fence["state"]
                is ManagedAsyncSaveState.SAVE_STAGING
            ):
                # MCore 0.17 calls inner state_dict while synchronously
                # constructing the AsyncRequest.  This is a read by the save
                # owner, before scheduling; waiting here would self-deadlock.
                self.drain()
                self._validate_bound_state_views()
            else:
                self.prepare_checkpoint_save()
        return super().state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load without torch Optimizer's automatic state-to-parameter-device cast."""
        self._wait_for_async_checkpoint_mutation("load_state_dict")
        if not self._bound:
            self._load_unbound_metadata_state_dict(state_dict)
            return
        self._ensure_checkpoint_state_mutation_allowed("load_state_dict")
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
                destination = self.state[param][key]
                value = loaded_state[key]
                destination.copy_(value)
                if (
                    self._checkpoint_lifecycle is _CheckpointLifecycle.LOAD_ACTIVE
                    and destination.data_ptr() != value.data_ptr()
                ):
                    self._checkpoint_loaded_state[param].add(key)

    def _ensure_checkpoint_state_mutation_allowed(self, operation: str) -> None:
        """Reject state writes while a retained rollback is the authority."""
        if (
            self._checkpoint_commit_is_decided()
            or self._checkpoint_cleanup is not None
            or self._checkpoint_lifecycle
            in (
                _CheckpointLifecycle.COMMIT_DECIDED,
                _CheckpointLifecycle.CLEANUP_PENDING,
            )
        ):
            raise RuntimeError(
                f"{operation} is unavailable while checkpoint cleanup is pending"
            )
        if self._checkpoint_rollback is not None and (
            self._checkpoint_lifecycle is not _CheckpointLifecycle.LOAD_ACTIVE
        ):
            raise RuntimeError(
                f"{operation} is unavailable while checkpoint rollback is pending"
            ) from self._checkpoint_load_error
        if self._checkpoint_lifecycle in (
            _CheckpointLifecycle.RECOVERY_PENDING,
            _CheckpointLifecycle.POISONED,
            _CheckpointLifecycle.RELOAD_REQUIRED,
        ):
            raise RuntimeError(
                f"{operation} requires an active full checkpoint recovery"
            ) from self._checkpoint_load_error

    def _load_unbound_metadata_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Support MCore's metadata-only DistributedOptimizer construction step."""
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
        actual_top = set(state_dict)
        if actual_top != expected_top:
            raise KeyError(
                "optimizer checkpoint top-level fields mismatch: "
                f"missing={sorted(expected_top - actual_top)}, "
                f"extra={sorted(actual_top - expected_top)}"
            )
        if self.cpu_slabs is None:
            raise RuntimeError("GPU-staged AdamW must be bound before state load")
        loaded_groups_value = state_dict["param_groups"]
        if not isinstance(loaded_groups_value, (list, tuple)):
            raise TypeError("optimizer param_groups must be a list or tuple")
        loaded_groups = list(loaded_groups_value)
        if len(loaded_groups) != len(self.param_groups):
            raise ValueError(
                "loaded optimizer has a different number of parameter groups"
            )
        id_to_param: dict[int, torch.Tensor] = {}
        seen_params: set[int] = set()
        for group_index, (current_group, loaded_group) in enumerate(
            zip(self.param_groups, loaded_groups)
        ):
            if not isinstance(loaded_group, Mapping):
                raise TypeError(
                    f"optimizer param_group {group_index} must be a mapping"
                )
            expected_group_keys = set(current_group)
            actual_group_keys = set(loaded_group)
            if actual_group_keys != expected_group_keys:
                raise KeyError(
                    f"optimizer param_group {group_index} fields mismatch: "
                    f"missing={sorted(expected_group_keys - actual_group_keys)}, "
                    f"extra={sorted(actual_group_keys - expected_group_keys)}"
                )
            loaded_ids = loaded_group["params"]
            if not isinstance(loaded_ids, (list, tuple)):
                raise TypeError(
                    f"optimizer param_group {group_index} params must be a sequence"
                )
            if len(loaded_ids) != len(current_group["params"]):
                raise ValueError("loaded optimizer parameter group size does not match")
            for loaded_id, param in zip(loaded_ids, current_group["params"]):
                if not isinstance(loaded_id, int) or isinstance(loaded_id, bool):
                    raise TypeError("optimizer parameter identifiers must be integers")
                if loaded_id in id_to_param:
                    raise ValueError(
                        f"optimizer parameter identifier {loaded_id} is duplicated"
                    )
                if id(param) in seen_params:
                    raise ValueError("owned parameter appears in more than one group")
                id_to_param[loaded_id] = param
                seen_params.add(id(param))
            self._validate_param_group_metadata(
                current_group, loaded_group, group_index
            )

        loaded_state_map = state_dict["state"]
        if not isinstance(loaded_state_map, Mapping):
            raise TypeError("optimizer state must be a mapping")
        if not loaded_state_map:
            raise ValueError("bound managed optimizer checkpoint state is empty")
        expected_ids = set(id_to_param)
        actual_ids = set(loaded_state_map)
        if actual_ids != expected_ids:
            raise ValueError("loaded optimizer state parameter set does not match")
        expected_state_keys = {"master_param", "exp_avg", "exp_avg_sq"}
        for loaded_id, loaded_state in loaded_state_map.items():
            if not isinstance(loaded_state, Mapping):
                raise TypeError(
                    f"optimizer state for parameter {loaded_id} must be a mapping"
                )
            actual_state_keys = set(loaded_state)
            if actual_state_keys != expected_state_keys:
                missing_fields = sorted(expected_state_keys - actual_state_keys)
                extra_fields = sorted(actual_state_keys - expected_state_keys)
                raise KeyError(
                    f"optimizer state for parameter {loaded_id} fields mismatch: "
                    f"missing {' '.join(missing_fields) or '<none>'}; "
                    f"extra {' '.join(extra_fields) or '<none>'}"
                )
            param = id_to_param[loaded_id]
            for key in expected_state_keys:
                value = loaded_state[key]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"loaded {key} must be a tensor")
                if value.device.type != "cpu":
                    raise RuntimeError(f"loaded {key} must be a CPU tensor")
                if value.dtype is not torch.float32:
                    raise TypeError(f"loaded {key} dtype must be torch.float32")
                destination = self.state[param].get(key)
                if not isinstance(destination, torch.Tensor):
                    raise RuntimeError(f"current optimizer state is missing {key}")
                if value.shape != destination.shape or value.numel() != param.numel():
                    raise ValueError(f"loaded {key} has an incompatible shape or numel")
        self._validate_bound_state_views()
        return loaded_groups, id_to_param

    @staticmethod
    def _validate_param_group_metadata(
        current: Mapping[str, Any], loaded: Mapping[str, Any], group_index: int
    ) -> None:
        from areal.engine.megatron_utils.gpu_staged_optimizer_checkpoint import (
            validate_managed_adamw_param_group,
        )

        validate_managed_adamw_param_group(
            loaded,
            current,
            location=f"optimizer param_group {group_index}",
            ignore_params=False,
        )

    def _validate_bound_state_views(self) -> None:
        assert self.cpu_slabs is not None
        slab_by_key = {
            "master_param": self.cpu_slabs.master,
            "exp_avg": self.cpu_slabs.exp_avg,
            "exp_avg_sq": self.cpu_slabs.exp_avg_sq,
        }
        for layout in self._layouts:
            state = self.state.get(layout.param)
            if not isinstance(state, Mapping) or set(state) != set(slab_by_key):
                raise RuntimeError(
                    "managed optimizer state schema no longer matches slabs"
                )
            for key, slab in slab_by_key.items():
                value = state[key]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.device.type != "cpu"
                    or value.dtype is not torch.float32
                    or value.shape != layout.param.shape
                    or value.numel() != layout.numel
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
