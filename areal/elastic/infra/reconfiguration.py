# SPDX-License-Identifier: Apache-2.0

"Native reconfiguration requests, durable application, engine execution and process cleanup."

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import re
import subprocess
import time
import uuid
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import psutil

from areal.elastic.contracts import ControlBusy, ControlSnapshot
from areal.elastic.infra.monitor.collector import ElasticMonitor
from areal.elastic.layout import ExecutionLayout as ExecutionLayout
from areal.elastic.layout import ReconfigurationPlan, ReconfigurationPlanner
from areal.utils.atomic import (
    artifact_manifest,
    atomic_json,
    json_digest,
    recovery_digest,
)

OWNER_ENV = "AREAL_ELASTIC_PROCESS_OWNER"


def owned_processes(owner: str) -> list[psutil.Process]:
    if not owner:
        raise ValueError("Process owner token must be nonempty")
    found = []
    for process in psutil.process_iter():
        try:
            if (
                process.status() != psutil.STATUS_ZOMBIE
                and process.environ().get(OWNER_ENV) == owner
            ):
                found.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def stop_owned_processes(
    owner: str, *, grace: float = 10, force_timeout: float = 10
) -> None:
    """Signal only processes carrying this launch's random token; fence PID reuse.

    Poll again for late descendants instead of treating leader exit as group exit.
    psutil's signal methods check process identity (PID and creation time).
    """
    for operation, timeout in (("terminate", grace), ("kill", force_timeout)):
        deadline = time.monotonic() + timeout
        signalled = set()
        while processes := owned_processes(owner):
            for process in processes:
                try:
                    identity = (process.pid, process.create_time())
                    if identity not in signalled:
                        getattr(process, operation)()
                        signalled.add(identity)
                except psutil.NoSuchProcess:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        if not owned_processes(owner):
            return
    raise RuntimeError(
        f"Owned processes remain alive: {[p.pid for p in owned_processes(owner)]}"
    )


def gpu_processes() -> dict[str, list[int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    processes: dict[str, list[int]] = {}
    for line in result.stdout.splitlines():
        gpu, pid = (part.strip() for part in line.split(","))
        processes.setdefault(gpu, []).append(int(pid))
    return processes


def resolve_gpu_uuids(devices: list[int]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    mapping = {
        int(index.strip()): uuid.strip()
        for index, uuid in (line.split(",") for line in result.stdout.splitlines())
    }
    return [mapping[index] for index in devices]


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
                "queue": owner.data_plane_id
                if hasattr(owner, "data_plane_id")
                else owner.inbox.queue.queue_id,
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
        if request_id in application.state().get("legacy_receipts", {}):
            raise ValueError("Request ID belongs to the immutable legacy Actor journal")
        application.preflight(request)
        if request_id in application.state()["receipts"]:
            return application.apply(request)
        plan = self.owner.plan_reconfiguration(target)
        self.owner.last_reconfiguration_plan = plan
        state = application.state()
        if self.owner.managed_rollout and (not state.get("managed_rollout", False)):
            atomic_json(
                application.root / "state.json", {**state, "managed_rollout": True}
            )
        if plan.steps:
            with self.owner._reconfiguration_phase("checkpoint"):
                self.owner.save_checkpoint(reason="resource-reconfigure")
            if self.owner.checkpoint_version != self.owner.applied_version:
                raise RuntimeError("Reconfiguration source checkpoint is not current")
        with self.owner._reconfiguration_phase("application_and_commit"):
            receipt = application.apply(request, plan=plan)
        if not self.owner.control_snapshot().boundary_ready:
            raise RuntimeError("Reconfiguration left an incomplete boundary")
        if receipt["status"] == "unchanged":
            return receipt
        arm = getattr(self.owner, "_arm_post_reconfiguration", None)
        if arm is not None:
            arm()
        with self.owner._reconfiguration_phase("generation_resume"):
            self.owner.rollout.continue_generation()
        with self.owner._reconfiguration_phase("admission_open"):
            self._set_admission(True)
            if self.owner.managed_rollout:
                self.owner.rollout.open_model_calls()
            else:
                self.owner.rollout.resume()
        return receipt

    def _set_admission(self, opened: bool) -> None:
        if hasattr(self.owner, "set_admission_open"):
            self.owner.set_admission_open(opened)
        else:
            self.owner.inbox.set_admission_open(opened)

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
                with self.owner._reconfiguration_phase("drain"):
                    self.owner.rollout.drain_model_calls()
            elif step == "close_admission":
                self._set_admission(False)
            elif step == "release_actor":
                with self.owner._reconfiguration_phase("actor_release"):
                    self.owner._release()
                    if self.owner.reconfigurator_adapter is not None:
                        self.owner.reconfigurator_adapter.role_released("actor")
            elif step == "disconnect_weights":
                with self.owner._reconfiguration_phase("awex_disconnect"):
                    self.owner.actor.disconnect_engine()
            elif step == "stop_rollout":
                with self.owner._reconfiguration_phase("rollout_stop"):
                    self.owner.rollout.stop_inference_replicas()
                    if self.owner.reconfigurator_adapter is not None:
                        self.owner.reconfigurator_adapter.role_released("rollout")
            elif step == "set_layout":
                with self.owner._reconfiguration_phase("placement"):
                    self.owner._set_execution_layout(plan.target)
            elif step == "rebuild_rollout":
                self.owner.rollout.rebuild_inference_replicas(
                    plan.target.rollout_devices,
                    timing=self.owner._reconfiguration_phase,
                )
            elif step == "rebind_runtime":
                if self.owner.reconfigurator_adapter is not None:
                    self.owner.monitor = ElasticMonitor(self.owner)
                    self.owner.reconfigurator_adapter.rebind_actor_layout(self.owner)
            elif step == "restore_actor":
                self.owner.ensure_actor(connect_weights=False)
            elif step == "connect_weights":
                with self.owner._reconfiguration_phase("awex_connect"):
                    self.owner.actor.connect_engine(
                        self.owner.rollout, self.owner.weight_meta
                    )
            elif step == "pause_rollout":
                self.owner.rollout.pause()
            elif step == "publish_weights":
                with self.owner._reconfiguration_phase("publish"):
                    self.owner._publish()
            else:
                raise ValueError(f"Unknown transition step: {step}")
        self.owner._reconfiguration_incomplete = False
        self.owner._checkpoint_dirty = True
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
        self.config = trainer.config
        self.options = self.config.actor_lifecycle
        self.root = Path(self.options.state_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.actor_devices = tuple(self.options.actor_devices)
        self.rollout_devices = tuple(self.options.rollout_devices)
        self.resource_reservation = self.actor_devices + self.rollout_devices
        uuids = resolve_gpu_uuids(list(self.resource_reservation))
        processes = gpu_processes()
        if any(processes.get(gpu) for gpu in uuids):
            raise RuntimeError("The explicitly assigned GPU pools are not empty")
        self.data_plane_id = uuid.uuid4().hex
        self.reconfiguration_planner = ReconfigurationPlanner()
        self.last_reconfiguration_plan = None
        self.last_checkpoint_reports = self.last_restore_reports = (
            self.last_restore_plan
        ) = None
        self.generation = 1
        self._reconfiguration_incomplete = False
        self._checkpoint_dirty = False
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
        self.reconfiguration_application, _ = configuration_application(self)

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
        self.trainer.actor.destroy()
        uuids = resolve_gpu_uuids(list(self.actor_devices))
        processes = gpu_processes()
        if any(processes.get(gpu) for gpu in uuids):
            raise RuntimeError("Actor GPU release is not confirmed")
        self.trainer.actor = None

    def ensure_actor(self, *, connect_weights=True):
        if self.actor is not None:
            return self.actor
        from areal.v2.training_service.dp_restore import DPStateRestorePlan

        config = deepcopy(self.config.actor)
        config.backend = f"fsdp:d{self.actor_world_size}"
        self.trainer._initialize_reconfigured_actor(config, self.ft_spec)
        source_dp = self.latest_manifest["world_size"]
        restore = {}
        self.last_restore_plan = None
        if source_dp != self.actor_world_size:
            self.last_restore_plan = asdict(
                DPStateRestorePlan(
                    identity_sha256=self.identity,
                    checkpoint_generation=self.latest_path.name,
                    policy_version=self.checkpoint_version,
                    source_world_size=source_dp,
                    target_world_size=self.actor_world_size,
                    transition_id=uuid.uuid4().hex + uuid.uuid4().hex,
                )
            )
            restore["restore_plan"] = self.last_restore_plan
        self.last_restore_reports = self.actor.full_checkpoint(
            "load",
            path=str(self.latest_path),
            backend=self.allocation.backend,
            identity_sha256=self.identity,
            version=self.checkpoint_version,
            **restore,
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

    def _reconfiguration_phase(self, name):
        return nullcontext()


def configuration_application(owner):
    """Open one journal, importing an earlier DP-only journal without changing it."""

    application = ConfigApplication(
        ReconfigurationExecutor(owner), owner.root / "resource-reconfiguration"
    )
    path = application.root / "state.json"
    legacy_path = owner.root / "actor-reconfiguration" / "state.json"
    existed = path.exists()
    if not existed and legacy_path.exists():
        state = json.loads(legacy_path.read_text())
        expected = json_digest(
            {
                "scope": "native-fsdp-dp-v1",
                "run": owner.identity,
                "queue": owner.inbox.queue.queue_id,
                "actor_gpu_pool": list(owner.options.actor_devices),
                "rollout_gpu_pool": list(owner.options.rollout_devices),
            }
        )
        if state["context_id"] != expected:
            raise ValueError("Actor journal belongs to another execution context")

        def layout(value):
            return asdict(
                ExecutionLayout(
                    value["world_size"],
                    tuple(owner.options.actor_devices),
                    tuple(owner.options.rollout_devices),
                )
            )

        def convert(row):
            row["request"]["target"] = layout(row["request"]["target"])
            for key in ("source", "actual"):
                if key in row:
                    row[key]["config"] = layout(row[key]["config"])

        state["config"] = layout(state["config"])
        for receipt in state["receipts"].values():
            convert(receipt)
        if state["pending"] is not None:
            convert(state["pending"])
        state["context_id"] = application.executor.context_id
        state["migrated_from"] = "actor-reconfiguration"
        state["managed_rollout"] = owner._managed_rollout
        atomic_json(path, state)
    if not path.exists():
        # New journals initialize after bind reconciles checkpoint/publication state.
        return application, {
            "config": asdict(owner.execution_layout),
            "pending": None,
            "managed_rollout": owner._managed_rollout,
        }
    state = application.state()
    if "managed_rollout" not in state:
        state["managed_rollout"] = True if existed else owner._managed_rollout
        atomic_json(path, state)
    return application, state


def submit_request(
    state_root: Path,
    *,
    target: ExecutionLayout,
    request_id: str,
    expected_generation: int,
    expected_weights_version: int,
) -> Path:
    if not (state_root / "state.json").is_file():
        raise ValueError("State root does not belong to an initialized native runtime")
    request = ConfigRequest(
        request_id, target, expected_generation, expected_weights_version, "manual"
    )
    root = state_root / "resource-reconfiguration"
    root.mkdir(exist_ok=True)
    with (root / "request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "request.json"
        if path.exists():
            raise RuntimeError("A reconfiguration request is already pending")
        atomic_json(path, asdict(request))
    return path


def poll_request(owner) -> None:
    root = owner.root / "resource-reconfiguration"
    if not (root / "request.json").is_file():
        return
    with (root / "request.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = json.loads((root / "request.json").read_text())
        try:
            if request.pop("reconfigurator_bundle_id") != "manual":
                raise ValueError("Reconfiguration mailbox requires manual provenance")
            target = ExecutionLayout.decode(request.pop("target"))
            response = {
                "request_id": request["request_id"],
                "receipt": owner.reconfigure(target=target, **request),
            }
        except Exception as error:
            response = {
                "request_id": request.get("request_id"),
                "error": f"{type(error).__name__}: {error}",
            }
            atomic_json(root / "response.json", response)
            application = owner.reconfiguration_application
            if (
                application is not None
                and application.state().get("pending") is not None
            ):
                raise
        atomic_json(root / "response.json", response)
        (root / "request.json").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    request = sub.add_parser("request")
    request.add_argument("--actor-devices", type=int, nargs="+", required=True)
    request.add_argument("--rollout-devices", type=int, nargs="+", required=True)
    request.add_argument("--world-size", type=int, required=True)
    request.add_argument("--request-id", required=True)
    request.add_argument("--expected-generation", type=int, required=True)
    request.add_argument("--expected-weights-version", type=int, required=True)
    args = parser.parse_args()
    if args.command == "status":
        status = {
            name: json.loads(path.read_text()) if path.exists() else None
            for name in ("state", "request", "response")
            for path in [args.state_root / "resource-reconfiguration" / f"{name}.json"]
        }
        status["authority"] = "resource" if status["state"] is not None else "actor"
        if status["state"] is None:
            legacy = args.state_root / "actor-reconfiguration/state.json"
            status["state"] = (
                json.loads(legacy.read_text()) if legacy.exists() else {"generation": 0}
            )
        print(json.dumps(status, indent=2))
    else:
        print(
            submit_request(
                args.state_root,
                target=ExecutionLayout(
                    args.world_size,
                    tuple(args.actor_devices),
                    tuple(args.rollout_devices),
                ),
                request_id=args.request_id,
                expected_generation=args.expected_generation,
                expected_weights_version=args.expected_weights_version,
            )
        )


if __name__ == "__main__":
    main()
