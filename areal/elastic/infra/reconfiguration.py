# SPDX-License-Identifier: Apache-2.0

"""Native reconfiguration requests, application journal and layout orchestration."""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import re
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

from areal.elastic.contracts import ControlBusy, ControlSnapshot
from areal.elastic.layout import ExecutionLayout as ExecutionLayout
from areal.elastic.layout import ReconfigurationPlan, ReconfigurationPlanner
from areal.elastic.reconfigurator import validate_reconfigurator
from areal.infra.utils.process import gpu_processes, resolve_gpu_uuids
from areal.utils.atomic import (
    artifact_manifest,
    atomic_json,
    json_digest,
    recovery_digest,
)


def validate_config(config: Any) -> None:
    """Reject unsupported state/placement combinations before starting services."""
    from areal.api.alloc_mode import ModelAllocation
    from areal.utils.environ import is_single_controller

    options = config.elastic
    validate_reconfigurator(options.reconfigurator)
    actor = ModelAllocation.from_str(config.actor.backend)
    rollout = ModelAllocation.from_str(config.rollout.backend)
    if not is_single_controller() or config.scheduler.type != "local":
        raise ValueError(
            "elastic reconfiguration currently requires a local single CPU coordinator"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError(
            "Start the elastic reconfiguration CPU coordinator with CUDA_VISIBLE_DEVICES empty"
        )
    if config.actor._version != "v2" or config.rollout._version != "v2":
        raise ValueError(
            "elastic reconfiguration requires native v2 Actor and Rollout controllers"
        )
    if actor.backend not in ("fsdp", "megatron") or rollout.backend != "sglang":
        raise ValueError(
            "elastic reconfiguration currently supports FSDP/Megatron with SGLang"
        )
    if config.rollout.agent is None or config.rollout.agent.mode != "online":
        raise ValueError("elastic reconfiguration requires agent.mode='online'")
    if (
        config.critic is not None
        or config.ref is not None
        or config.teacher is not None
        or config.actor.use_lora
    ):
        raise ValueError(
            "elastic reconfiguration does not yet support critic/ref/teacher or LoRA state"
        )
    if (
        config.enable_offload
        or config.actor.offload
        or config.rollout.api_url is not None
    ):
        raise ValueError(
            "elastic reconfiguration requires local separated workers without offloading"
        )
    for engine in (config.actor, config.rollout):
        strategy = engine.scheduling_strategy
        if strategy is not None and strategy.type != "separation":
            raise ValueError("elastic reconfiguration requires separation scheduling")
        if any(
            "CUDA_VISIBLE_DEVICES" in spec.env_vars for spec in engine.scheduling_spec
        ):
            raise ValueError(
                "Configure elastic reconfiguration GPU pools instead of scheduling env CUDA_VISIBLE_DEVICES"
            )
        if len(engine.scheduling_spec) != 1:
            raise ValueError(
                "elastic reconfiguration requires exactly one scheduling spec per engine"
            )
    if (
        config.actor.scheduling_spec[0].gpu != 1
        or config.rollout.scheduling_spec[0].gpu <= 0
    ):
        raise ValueError(
            "elastic reconfiguration requires one GPU per Actor guard and a positive Rollout GPU request"
        )
    actor_size = actor.parallel.world_size
    actor_pool = len(options.actor_devices)
    pure_fsdp_dp = actor.backend == "fsdp" and all(
        getattr(actor.parallel, name) == 1
        for name in ("tp_size", "pp_size", "cp_size", "ep_size", "etp_size")
    )
    if (
        rollout.parallel.world_size != len(options.rollout_devices)
        or actor_size > actor_pool
        or actor_size != actor_pool
        and not pure_fsdp_dp
    ):
        raise ValueError(
            "GPU pools must match their layouts; pure FSDP DP may use a subset of its reserved pool"
        )
    if (
        config.gconfig.n_samples != 1
        or config.gconfig.reward_normalization
        or config.dynamic_bs
    ):
        raise ValueError(
            "elastic reconfiguration currently requires independent single-sample episodes and fixed batches"
        )
    if config.rollout.consumer_batch_size != config.train_dataset.batch_size:
        raise ValueError(
            "Rollout consumer_batch_size must equal the independent episode batch size"
        )
    for norm in (config.actor.adv_norm, config.actor.reward_norm):
        if norm is not None and (
            norm.mean_level == "group" or norm.std_level == "group"
        ):
            raise ValueError(
                "Independent single-sample episodes cannot use group normalization"
            )
    if not Path(config.actor.path).is_dir():
        raise ValueError(
            "elastic reconfiguration requires a pinned, locally materialized base model"
        )
    if Path(config.sglang.model_path).resolve() != Path(config.actor.path).resolve():
        raise ValueError("Actor and SGLang must start from the same pinned base model")


ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class ConfigRequest(Generic[ConfigT]):
    request_id: str
    target: ConfigT
    expected_generation: int
    expected_weights_version: int
    reconfigurator_bundle_id: str

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", self.request_id
        ):
            raise ValueError("Invalid config request identity")
        if not is_dataclass(self.target) or isinstance(self.target, type):
            raise TypeError(
                "Target must be a complete execution configuration dataclass"
            )
        if any(
            type(n) is not int or n < 0
            for n in (self.expected_generation, self.expected_weights_version)
        ):
            raise ValueError("Expected generations/versions must be nonnegative")
        if (
            not isinstance(self.reconfigurator_bundle_id, str)
            or not self.reconfigurator_bundle_id
        ):
            raise ValueError("Scheduling reconfigurator provenance is required")


@dataclass(frozen=True)
class ConfigSnapshot(Generic[ConfigT]):
    config: ConfigT
    weights_version: int
    boundary_id: str
    committed: bool


class ConfigExecutor(Protocol[ConfigT]):
    """Implementations must verify all changed roles, not just assign fields."""

    @property
    def context_id(self) -> str: ...

    def observe(self) -> ConfigSnapshot[ConfigT]: ...

    def decode_config(self, value: dict) -> ConfigT:
        """Reconstruct this executor's schema without importing planning models."""
        ...

    def validate(self, target: ConfigT) -> None: ...

    def apply(self, target: ConfigT, *, plan=None) -> None: ...

    def recover(self, source: ConfigSnapshot[ConfigT]) -> None: ...


class ConfigApplication(Generic[ConfigT]):
    """One application journal; partial requests recover to their source boundary.

    Config generations count committed logical changes, not training updates.
    Reusing an id returns its historical receipt, even after later transitions.
    Source/target capabilities and logical training semantics belong to the executor.
    """

    def __init__(self, executor: ConfigExecutor[ConfigT], root: Path):
        self.executor = executor
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict:
        path = self.root / "state.json"
        if not path.exists():
            current = self.executor.observe()
            if not current.committed:
                raise RuntimeError("Cannot initialize from uncommitted execution")
            atomic_json(
                path,
                {
                    "format_version": 1,
                    "context_id": self.executor.context_id,
                    "generation": 0,
                    "config": asdict(current.config),
                    "pending": None,
                    "receipts": {},
                },
            )
        state = json.loads(path.read_text())
        if (
            state["format_version"] != 1
            or state["context_id"] != self.executor.context_id
        ):
            raise ValueError("Config journal belongs to another execution context")
        return state

    def state(self) -> dict:
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._read()

    def _preflight(self, state: dict, request: ConfigRequest[ConfigT]):
        if state["pending"] is not None:
            raise RuntimeError("Recover the pending config application first")
        current = self.executor.observe()
        if (
            not current.committed
            or json.loads(json.dumps(asdict(current.config))) != state["config"]
        ):
            raise RuntimeError("Execution differs from the committed config journal")
        old = state["receipts"].get(request.request_id)
        if old is not None:
            if old["request"] != json.loads(json.dumps(asdict(request))):
                raise ValueError(
                    "Config request identity was reused with different input"
                )
            return old
        if (
            request.expected_generation != state["generation"]
            or request.expected_weights_version != current.weights_version
        ):
            raise ValueError("Stale config generation or training weight version")
        self.executor.validate(request.target)
        return None

    def preflight(self, request: ConfigRequest[ConfigT]) -> None:
        """Check before closing admission; apply repeats checks under the same gate."""
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._preflight(self._read(), request)

    def _verify(
        self, source: ConfigSnapshot[ConfigT], target: ConfigT
    ) -> ConfigSnapshot[ConfigT]:
        actual = self.executor.observe()
        if (
            not actual.committed
            or actual.config != target
            or actual.weights_version != source.weights_version
            or actual.boundary_id != source.boundary_id
        ):
            raise RuntimeError(
                "Config application changed progress or failed verification"
            )
        return actual

    def _finish(self, state, pending, actual, status, error=None):
        if status == "applied":
            state["generation"] += 1
        receipt = {
            "request": pending["request"],
            "status": status,
            "generation": state["generation"],
            "actual": json.loads(json.dumps(asdict(actual))),
            "seconds": max(0, time.time() - pending["started"]),
            "error": error,
        }
        if "plan" in pending:
            receipt["plan"] = pending["plan"]
        state["config"] = asdict(actual.config)
        state["receipts"][pending["request"]["request_id"]] = receipt
        state["pending"] = None
        atomic_json(self.root / "state.json", state)
        return receipt

    def apply(self, request: ConfigRequest[ConfigT], *, plan=None) -> dict:
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = self._read()
            previous = self._preflight(state, request)
            if previous is not None:
                return previous
            source = self.executor.observe()
            if plan is not None and (
                plan.source != source.config or plan.target != request.target
            ):
                raise ValueError(
                    "Execution plan differs from the source or target layout"
                )
            pending = {
                "request": json.loads(json.dumps(asdict(request))),
                "source": json.loads(json.dumps(asdict(source))),
                "phase": "applying",
                "started": time.time(),
            }
            if plan is not None:
                pending["plan"] = json.loads(json.dumps(asdict(plan)))
            state["pending"] = pending
            atomic_json(self.root / "state.json", state)
            if request.target == source.config:
                return self._finish(state, pending, source, "unchanged")
            try:
                if plan is None:
                    self.executor.apply(request.target)
                else:
                    self.executor.apply(request.target, plan=plan)
                actual = self._verify(source, request.target)
            except Exception as error:
                pending["phase"] = "recovering"
                pending["error"] = f"{type(error).__name__}: {error}"
                atomic_json(self.root / "state.json", state)
                self.executor.recover(source)
                actual = self._verify(source, source.config)
                return self._finish(
                    state, pending, actual, "rolled_back", pending["error"]
                )
            return self._finish(state, pending, actual, "applied")

    def recover(self) -> dict | None:
        """Reconcile a partial request, or restore active config on a fresh trainer."""
        with (self.root / "config.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = self._read()
            pending = state["pending"]
            if pending is not None:
                row = pending["source"]
                source = ConfigSnapshot(
                    self.executor.decode_config(row["config"]),
                    row["weights_version"],
                    row["boundary_id"],
                    row["committed"],
                )
                self.executor.recover(source)
                actual = self._verify(source, source.config)
                return self._finish(
                    state, pending, actual, "rolled_back", "Interrupted application"
                )
            current = self.executor.observe()
            if not current.committed:
                raise RuntimeError(
                    "Recover training progress before config application"
                )
            active = self.executor.decode_config(state["config"])
            if current.config != active:
                source = replace(current, config=active)
                self.executor.recover(source)
                self._verify(source, active)
            return None


class ReconfigurationExecutor:
    def __init__(self, owner):
        self.owner = owner
        self.context_id = json_digest(
            {
                "scope": "native-resources-v1",
                "run": owner.identity,
                "data_plane": owner.data_plane_id,
                "reservation": list(owner.resource_reservation),
            }
        )

    def observe(self):
        state = self.owner.control_snapshot()
        return ConfigSnapshot(
            self.owner.execution_layout,
            state.applied_version,
            state.boundary_id,
            state.boundary_ready,
        )

    def decode_config(self, value):
        return ExecutionLayout.decode(value)

    def validate(self, target):
        self.owner.validate_execution_layout(target)

    def apply(self, target, *, plan: ReconfigurationPlan | None = None):
        plan = plan or self.owner.plan_reconfiguration(target)
        if plan.target != target:
            raise ValueError("Execution plan has a different target")
        self.execute(plan)

    def recover(self, source):
        if self.observe() != source:
            self.recover_layout(source.config)
        if self.observe() != source:
            raise RuntimeError(
                "Reconfiguration recovery did not preserve the source training boundary"
            )

    def reconfigure(
        self,
        *,
        target: ExecutionLayout,
        request_id: str,
        expected_generation: int,
        expected_weights_version: int,
        reconfigurator_bundle_id: str = "manual",
    ):
        request = ConfigRequest(
            request_id,
            target,
            expected_generation,
            expected_weights_version,
            reconfigurator_bundle_id,
        )
        self.owner.validate_execution_layout(target)
        if not self.owner.control_snapshot().boundary_ready:
            raise ControlBusy("Reconfiguration requires a completed, published update")
        if self.owner.applied_version < 1:
            raise ValueError("Reconfiguration requires a completed training update")
        application = self.owner.reconfiguration_application
        application.preflight(request)
        if request_id in application.state()["receipts"]:
            return application.apply(request)
        plan = self.owner.plan_reconfiguration(target)
        self.owner.last_reconfiguration_plan = plan
        if plan.steps:
            self.owner.save_checkpoint(reason="resource-reconfigure")
            if self.owner.checkpoint_version != self.owner.applied_version:
                raise RuntimeError("Reconfiguration source checkpoint is not current")
        receipt = application.apply(request, plan=plan)
        if not self.owner.control_snapshot().boundary_ready:
            raise RuntimeError("Reconfiguration left an incomplete boundary")
        if receipt["status"] == "unchanged":
            return receipt
        self.owner.rollout.continue_generation()
        self.owner.set_admission_open(True)
        if self.owner.managed_rollout:
            self.owner.rollout.open_model_calls()
        else:
            self.owner.rollout.resume()
        return receipt

    def execute(self, plan: ReconfigurationPlan) -> None:
        """Execute an already selected sequence; unchanged engines stay alive."""
        if plan.source != self.owner.execution_layout or plan.actor_running != (
            self.owner.actor is not None
        ):
            raise ControlBusy("The source state changed after transition planning")
        if not plan.steps:
            return
        if (
            self.owner.active is not None
            or self.owner.checkpoint_version != self.owner.applied_version
        ):
            raise RuntimeError("Reconfiguration requires the saved completed update")
        self.owner._reconfiguration_incomplete = True
        self.owner._event(
            "RECONFIGURING",
            resource_target=dataclasses.asdict(plan.target),
            plan=dataclasses.asdict(plan),
        )
        for step in plan.steps:
            if step == "drain_model_calls":
                self.owner.rollout.drain_model_calls()
            elif step == "close_admission":
                self.owner.set_admission_open(False)
            elif step == "release_actor":
                self.owner._release()
                if self.owner.reconfigurator_adapter is not None:
                    self.owner.reconfigurator_adapter.role_released("actor")
            elif step == "disconnect_weights":
                self.owner.actor.disconnect_engine()
            elif step == "stop_rollout":
                self.owner.rollout.stop_inference_replicas()
                if self.owner.reconfigurator_adapter is not None:
                    self.owner.reconfigurator_adapter.role_released("rollout")
            elif step == "set_layout":
                self.owner._set_execution_layout(plan.target)
            elif step == "rebuild_rollout":
                self.owner.rollout.rebuild_inference_replicas(
                    plan.target.rollout_devices
                )
            elif step == "rebind_runtime":
                if self.owner.reconfigurator_adapter is not None:
                    self.owner.reconfigurator_adapter.rebind_actor_layout(self.owner)
            elif step == "restore_actor":
                self.owner.ensure_actor(connect_weights=False)
            elif step == "connect_weights":
                self.owner.actor.connect_engine(
                    self.owner.rollout, self.owner.weight_meta
                )
            elif step == "pause_rollout":
                self.owner.rollout.pause()
            elif step == "publish_weights":
                self.owner._publish()
            else:
                raise ValueError(f"Unknown transition step: {step}")
        self.owner._reconfiguration_incomplete = False
        self.owner._event(
            "RECONFIGURED", serving_versions=self.owner.rollout.serving_versions()
        )

    def recover_layout(self, target: ExecutionLayout) -> None:
        plan = self.owner.reconfiguration_planner.plan(
            self.owner.execution_layout,
            target,
            actor_running=self.owner.actor is not None,
            force_actor=True,
            force_rollout=self.owner.managed_rollout,
        )
        self.execute(plan)


class TrainerReconfiguration:
    """Execute layout changes against Trainer-owned engines and live data services.

    This object owns the transition journal and full switch checkpoints only.
    It neither consumes batches nor advances training, publication or save schedules.
    CPU coordinator restart/replay is outside this execution contract.
    """

    reconfigurator_adapter = None

    def __init__(self, trainer, ft_spec):
        self.trainer, self.ft_spec = trainer, ft_spec
        self.monitor = None
        self.config = trainer.config
        self.options = self.config.elastic
        self.root = Path(self.options.state_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.actor_devices = tuple(self.options.actor_devices)
        self.rollout_devices = tuple(self.options.rollout_devices)
        self.resource_reservation = self.actor_devices + self.rollout_devices
        uuids = resolve_gpu_uuids(list(self.resource_reservation))
        self.gpu_uuids = tuple(uuids[: len(self.actor_devices)])
        processes = gpu_processes()
        if any(processes.get(gpu) for gpu in uuids):
            raise RuntimeError("The explicitly assigned GPU pools are not empty")
        self.data_plane_id = uuid.uuid4().hex
        self.reconfiguration_planner = ReconfigurationPlanner()
        self.last_reconfiguration_plan = None
        self.last_checkpoint_reports = self.last_restore_reports = None
        self.generation = 1
        self._reconfiguration_incomplete = False
        self._managed_rollout = (
            os.environ.get("AREAL_ELASTIC_RESOURCE_RECONFIGURATION") == "1"
        )
        self.latest_path = self.latest_manifest = None
        self.base_manifest = artifact_manifest(Path(self.config.actor.path))
        self.compatibility = json.loads(
            json.dumps(
                {
                    "native_full_state": 1,
                    "backend": trainer.actor_alloc.backend,
                    "actor": asdict(self.config.actor),
                    "base_files": self.base_manifest,
                    "ft_spec": asdict(ft_spec),
                    "seed": self.config.seed,
                    "batch_size": self.config.train_dataset.batch_size,
                    "max_staleness": self.config.rollout.max_head_offpolicyness,
                }
            )
        )
        self.identity = recovery_digest(self.compatibility)
        from areal.elastic.infra.state.checkpoint_store import CheckpointStore

        self.store = CheckpointStore(
            self.root / "checkpoints",
            keep=self.options.checkpoint_keep,
            free_space_floor=self.options.checkpoint_free_space_floor,
        )
        self.reconfiguration_application = ConfigApplication(
            ReconfigurationExecutor(self), self.root / "resource-reconfiguration"
        )

    @property
    def actor(self):
        return self.trainer.actor

    @property
    def rollout(self):
        return self.trainer.rollout

    @property
    def allocation(self):
        return self.trainer.actor_alloc

    @property
    def actor_world_size(self):
        return self.allocation.parallel.world_size

    @property
    def actor_backend(self):
        return self.actor.config.backend

    @property
    def weight_meta(self):
        return self.trainer.weight_update_meta

    @property
    def applied_version(self):
        return (
            self.actor.get_version()
            if self.actor is not None
            else self.checkpoint_version
        )

    @property
    def published_version(self):
        return self.rollout.get_version()

    @property
    def checkpoint_version(self):
        return self.latest_manifest["global_step"] if self.latest_manifest else 0

    @property
    def active(self):
        return True if self.trainer._training_active else None

    @property
    def execution_layout(self):
        return ExecutionLayout(
            self.actor_world_size, self.actor_devices, self.rollout_devices
        )

    @property
    def managed_rollout(self):
        return self._managed_rollout or (
            self.rollout is not None and self.rollout.resource_reconfiguration_enabled
        )

    @property
    def supports_resource_reconfiguration(self):
        from areal.api.alloc_mode import ModelAllocation

        actor = self.allocation.parallel
        rollout = ModelAllocation.from_str(self.config.rollout.backend)
        return (
            self.allocation.backend == "fsdp"
            and all(
                getattr(actor, key) == 1
                for key in ("tp_size", "pp_size", "cp_size", "ep_size", "etp_size")
            )
            and rollout.backend == "sglang"
            and rollout.parallel.tp_size == rollout.parallel.pp_size == 1
        )

    def control_snapshot(self):
        failed = self.trainer._training_failed
        version, published = self.applied_version, self.published_version
        return ControlSnapshot(
            False,
            not failed
            and self.active is None
            and not self._reconfiguration_incomplete
            and version == published,
            self.generation,
            version,
            published,
            json_digest({"data_plane": self.data_plane_id, "version": version}),
            checkpoint_version=self.checkpoint_version,
        )

    def validate_execution_layout(self, target):
        if target == self.execution_layout:
            return
        if not self.supports_resource_reconfiguration:
            raise ValueError(
                "Native layout changes require pure FSDP DP and SGLang TP1"
            )
        if not set(target.actor_devices + target.rollout_devices) <= set(
            self.resource_reservation
        ):
            raise ValueError("Target exceeds the original GPU reservation")
        if self.config.train_dataset.batch_size % target.world_size:
            raise ValueError("Actor DP must divide the logical training batch")
        if (
            target.actor_devices != self.actor_devices
            or target.rollout_devices != self.rollout_devices
        ) and not self.managed_rollout:
            raise ValueError("GPU pool changes require managed replicas")

    def plan_reconfiguration(self, target):
        self.validate_execution_layout(target)
        return self.reconfiguration_planner.plan(
            self.execution_layout, target, actor_running=self.actor is not None
        )

    def set_admission_open(self, opened):
        # Existing workflows and sessions retain their data throughout the switch.
        # Only new workflow submission is paused; model-call draining is a separate
        # layout-plan step when Rollout engines themselves are replaced.
        if opened:
            self.rollout.resume()
        else:
            self.rollout.pause()

    def _set_execution_layout(self, target):
        from areal.api.alloc_mode import ModelAllocation

        self.validate_execution_layout(target)
        if target.actor_devices != self.actor_devices:
            self.trainer.scheduler.set_gpu_pool(list(target.actor_devices))
        self.actor_devices, self.rollout_devices = (
            target.actor_devices,
            target.rollout_devices,
        )
        self.gpu_uuids = tuple(resolve_gpu_uuids(list(self.actor_devices)))
        self.trainer.actor_alloc = ModelAllocation.from_str(
            f"fsdp:d{target.world_size}", name="actor"
        )

    def save_checkpoint(self, *, reason):
        version = self.applied_version
        if not self.control_snapshot().boundary_ready:
            raise ControlBusy(
                "A switch checkpoint requires a completed published update"
            )
        backend = self.actor_backend

        def write(path):
            self.last_checkpoint_reports = self.actor.full_checkpoint(
                "save",
                path=str(path),
                backend=self.allocation.backend,
                identity_sha256=self.identity,
                version=version,
            )
            atomic_json(
                path / "actor-layout.json",
                {
                    "backend": backend,
                    "world_size": self.actor_world_size,
                    "format_version": 2,
                    "gpu_pool": list(self.actor_devices),
                    "rollout_devices": list(self.rollout_devices),
                    "reservation": list(self.resource_reservation),
                },
            )

        self.latest_path = self.store.commit(
            step=version,
            world_size=self.actor_world_size,
            compatibility=self.compatibility,
            write=write,
            estimated_bytes=self.options.checkpoint_estimated_bytes,
        )
        self.latest_path, self.latest_manifest = self.store.latest(self.compatibility)
        return True

    def _release(self):
        self.trainer.actor.destroy(release_devices=self.actor_devices)
        self.trainer.actor = None

    def ensure_actor(self, *, connect_weights=True):
        if self.actor is not None:
            return self.actor
        config = deepcopy(self.config.actor)
        config.backend = f"fsdp:d{self.actor_world_size}"
        self.trainer._initialize_reconfigured_actor(config, self.ft_spec)
        self.last_restore_reports = self.actor.full_checkpoint(
            "load",
            path=str(self.latest_path),
            backend=self.allocation.backend,
            identity_sha256=self.identity,
            version=self.checkpoint_version,
            source_world_size=self.latest_manifest["world_size"],
        )
        self.generation += 1
        if connect_weights:
            self.actor.connect_engine(self.rollout, self.weight_meta)
        return self.actor

    def _publish(self):
        version = self.applied_version
        self.actor.update_weights(
            self.weight_meta.with_version(version), resume_generation=False
        )
        versions = self.rollout.serving_versions()
        if not versions or any(value != version for value in versions):
            raise RuntimeError("Serving peers did not acknowledge the restored version")

    def _event(self, state, **details):
        atomic_json(
            self.root / "state.json",
            {
                "state": state,
                "generation": self.generation,
                "weights_version": self.published_version,
                "layout": asdict(self.execution_layout),
                **details,
            },
        )
