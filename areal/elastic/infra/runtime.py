# SPDX-License-Identifier: Apache-2.0

"""CPU-owned lifecycle around the native PPO Actor and persistent online inbox."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any

from areal.elastic.contracts import ControlBusy, ControlSnapshot
from areal.elastic.infra.adapter import ReconfiguratorAdapter, validate_unbound_adapter
from areal.elastic.infra.monitor.collector import ElasticMonitor, TransitionTiming
from areal.elastic.infra.reconfiguration import (
    OWNER_ENV,
    ExecutionLayout,
    configuration_application,
    gpu_processes,
    owned_processes,
    resolve_gpu_uuids,
    stop_owned_processes,
)
from areal.elastic.infra.reconfiguration import (
    poll_request as poll_reconfiguration_request,
)
from areal.elastic.infra.state.batch_queue import QueuedBatch
from areal.elastic.infra.state.checkpoint_store import CheckpointStore
from areal.elastic.infra.state.episode_inbox import OnlineInbox
from areal.elastic.layout import ReconfigurationPlan, ReconfigurationPlanner
from areal.elastic.reconfigurator import validate_reconfigurator
from areal.utils import logging
from areal.utils.atomic import (
    artifact_manifest,
    atomic_json,
    json_digest,
    recovery_digest,
)
from areal.utils.timeutil import EpochStepTimeFreqCtl

logger = logging.getLogger("RLTrainer")


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lifecycle_lock:
            return method(self, *args, **kwargs)

    return call


def _timed_reconfiguration(method):
    @wraps(method)
    def call(self, **kwargs):
        request_id = kwargs["request_id"]
        path = self.root / "resource-reconfiguration" / "state.json"
        if path.exists() and request_id in json.loads(path.read_text())["receipts"]:
            return method(self, **kwargs)
        self._record_post_reconfiguration()
        target = kwargs["target"]
        timing = TransitionTiming(
            self.root,
            request_id,
            {
                "layout": dataclasses.asdict(self.execution_layout),
                "weights_version": self.applied_version,
                "cursor": self.consumer.cursor.copy(),
            },
            dataclasses.asdict(target),
        )
        self._transition_timing = timing
        timing.record("begin", source=timing.source, target=timing.target)
        try:
            with timing.phase("total"):
                result = method(self, **kwargs)
            timing.record(
                "end", status=result["status"], generation=result["generation"]
            )
            return result
        except BaseException:
            timing.record("end", status="failed")
            raise
        finally:
            self._transition_timing = None

    return call


def validate_on_demand(config: Any) -> None:
    """Reject unsupported state/placement combinations before starting services."""
    from areal.api.alloc_mode import ModelAllocation
    from areal.utils.environ import is_single_controller

    lifecycle = config.actor_lifecycle
    validate_reconfigurator(lifecycle.reconfigurator)
    actor = ModelAllocation.from_str(config.actor.backend)
    rollout = ModelAllocation.from_str(config.rollout.backend)
    if not is_single_controller() or config.scheduler.type != "local":
        raise ValueError("on_demand currently requires a local single CPU coordinator")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError(
            "Start the on_demand CPU coordinator with CUDA_VISIBLE_DEVICES empty"
        )
    if config.actor._version != "v2" or config.rollout._version != "v2":
        raise ValueError("on_demand requires native v2 Actor and Rollout controllers")
    if actor.backend not in ("fsdp", "megatron") or rollout.backend != "sglang":
        raise ValueError("on_demand currently supports FSDP/Megatron with SGLang")
    if config.rollout.agent is None or config.rollout.agent.mode != "online":
        raise ValueError("on_demand requires agent.mode='online'")
    if (
        config.critic is not None
        or config.ref is not None
        or config.teacher is not None
        or config.actor.use_lora
    ):
        raise ValueError(
            "on_demand does not yet support critic/ref/teacher or LoRA state"
        )
    if (
        config.enable_offload
        or config.actor.offload
        or config.rollout.api_url is not None
    ):
        raise ValueError(
            "on_demand requires local separated workers without offloading"
        )
    for engine in (config.actor, config.rollout):
        strategy = engine.scheduling_strategy
        if strategy is not None and strategy.type != "separation":
            raise ValueError("on_demand requires separation scheduling")
        if any(
            "CUDA_VISIBLE_DEVICES" in spec.env_vars for spec in engine.scheduling_spec
        ):
            raise ValueError(
                "Configure on_demand GPU pools instead of scheduling env CUDA_VISIBLE_DEVICES"
            )
        if len(engine.scheduling_spec) != 1:
            raise ValueError(
                "on_demand requires exactly one scheduling spec per engine"
            )
    if (
        config.actor.scheduling_spec[0].gpu != 1
        or config.rollout.scheduling_spec[0].gpu <= 0
    ):
        raise ValueError(
            "on_demand requires one GPU per Actor guard and a positive Rollout GPU request"
        )
    actor_size = actor.parallel.world_size
    actor_pool = len(lifecycle.actor_devices)
    pure_fsdp_dp = actor.backend == "fsdp" and all(
        getattr(actor.parallel, name) == 1
        for name in ("tp_size", "pp_size", "cp_size", "ep_size", "etp_size")
    )
    if (
        rollout.parallel.world_size != len(lifecycle.rollout_devices)
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
            "on_demand currently requires independent single-sample episodes and fixed batches"
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
        raise ValueError("on_demand requires a pinned, locally materialized base model")
    if Path(config.sglang.model_path).resolve() != Path(config.actor.path).resolve():
        raise ValueError("Actor and SGLang must start from the same pinned base model")


class OnDemandActor:
    """Keep the consumer fence, serving service and data progress across GPU generations.

    The caller executes the ordinary native PPO operations between next_batch()
    and finish_update(). Recovery replays lost updates through the same PPO path.
    """

    def __init__(
        self,
        config: Any,
        ft_spec: Any,
        actor_factory: Callable[[Any], Any],
        *,
        reconfigurator_adapter: ReconfiguratorAdapter | None = None,
        actor_pool_setter: Callable[[list[int]], None] | None = None,
    ) -> None:
        validate_unbound_adapter(reconfigurator_adapter)
        self.config, self.ft_spec, self.actor_factory = config, ft_spec, actor_factory
        self._lifecycle_lock = threading.RLock()
        self.reconfigurator_adapter = (
            reconfigurator_adapter
            if reconfigurator_adapter is not None
            else ReconfiguratorAdapter()
        )
        self.revision_hooks = self.reconfigurator_adapter.hooks
        self._revision_hooks_bound = False
        self._train_started: float | None = None
        self.options = config.actor_lifecycle
        self.actor_devices = tuple(self.options.actor_devices)
        self.rollout_devices = tuple(self.options.rollout_devices)
        self.resource_reservation = self.actor_devices + self.rollout_devices
        self._actor_pool_setter = actor_pool_setter
        self.reconfiguration_application = None
        self.reconfiguration_planner = ReconfigurationPlanner()
        self.last_reconfiguration_plan = None
        self._managed_rollout = (
            os.environ.get("AREAL_ELASTIC_RESOURCE_RECONFIGURATION") == "1"
        )
        self._reconfiguration_incomplete = False
        self._transition_timing = self._pending_transition_timing = None
        self.last_restore_reports = self.last_restore_plan = None
        self.last_checkpoint_reports = None
        validate_reconfigurator(self.options.reconfigurator)
        self.checkpoint_frequency = EpochStepTimeFreqCtl(
            freq_epoch=config.recover.freq_epochs,
            freq_step=config.recover.freq_steps,
            freq_sec=config.recover.freq_secs,
        )
        self._initial_checkpoint_frequency = deepcopy(
            self.checkpoint_frequency.state_dict()
        )
        self.root = Path(self.options.state_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox = OnlineInbox(
            self.root / "inbox",
            batch_size=config.train_dataset.batch_size,
            max_staleness=config.rollout.max_head_offpolicyness,
        )
        from areal.api.alloc_mode import ModelAllocation

        self.allocation = ModelAllocation.from_str(config.actor.backend)
        self._actor_backend = config.actor.backend
        # Hash immutable model files, omitting download-cache bookkeeping.
        base = Path(config.actor.path)
        self.base_manifest = artifact_manifest(base)
        files = {
            name: digest
            for name, digest in self.base_manifest.items()
            if name.endswith((".safetensors", ".bin", ".index.json"))
            or name == "config.json"
        }
        if "config.json" not in files or not any(
            name.endswith((".safetensors", ".bin")) for name in files
        ):
            raise ValueError(
                "The pinned base model has no complete configuration/weights"
            )
        self.compatibility = {
            "native_full_state": 1,
            "backend": self.allocation.backend,
            "actor": dataclasses.asdict(config.actor),
            "base_files": files,
            "ft_spec": dataclasses.asdict(ft_spec),
            "seed": config.seed,
            "batch_size": self.inbox.batch_size,
            "max_staleness": self.inbox.max_staleness,
        }
        # Canonical JSON makes tuples/enums compare identically after recovery.
        self.compatibility = json.loads(json.dumps(self.compatibility))
        self.identity = recovery_digest(self.compatibility)
        self.store = CheckpointStore(
            self.root / "checkpoints",
            keep=self.options.checkpoint_keep,
            free_space_floor=self.options.checkpoint_free_space_floor,
        )
        self.latest_path: Path | None = None
        self.latest_manifest: dict[str, Any] | None = None
        self.actor: Any | None = None
        self.rollout: Any | None = None
        self.weight_meta: Any | None = None
        self.active: QueuedBatch | None = None
        self._update_applied = False
        self._publication_uncertain = False
        self._checkpoint_dirty = False
        self._active_data_progress = None
        self.owner: str | None = None
        self.ready_since: float | None = None
        self.idle_since: float | None = None
        self.started_at: float | None = None
        self.generation = 0
        self._closed = False
        self.monitor = ElasticMonitor(self)
        self._capture_cpu: Callable[[], dict[str, Any]] = lambda: {}
        self._restore_cpu: Callable[[dict[str, Any]], None] = lambda state: None
        self.consumer = self.inbox.queue.consumer(self._read_latest)
        try:
            self.inbox.set_admission_open(False)
            self.applied_version = (
                self.latest_manifest["global_step"] if self.latest_manifest else 0
            )
            self._restore_data_progress()
            previous = self.root / "state.json"
            if (self.root / "policy-application").exists():
                raise ValueError(
                    "Legacy policy-application state requires explicit migration; "
                    "use the original revision or a new state root"
                )
            if (
                not self.reconfigurator_adapter.supports_reconfigurator_activation
                and (self.root / "reconfigurator-application").exists()
            ):
                raise ValueError(
                    "Resume this coordinator with its AutoPilot RevisionAdapter"
                )
            if previous.exists():
                journal = json.loads(previous.read_text())
                self.generation = journal["generation"]
                if journal.get("owner"):
                    raise RuntimeError(
                        "Previous coordinator left an unconfirmed Actor generation; preserve its journal for cleanup"
                    )
            self.gpu_uuids = resolve_gpu_uuids(list(self.actor_devices))
            rollout_gpus = resolve_gpu_uuids(list(self.options.rollout_devices))
            present = gpu_processes()
            if any(present.get(gpu) for gpu in self.gpu_uuids + rollout_gpus):
                raise RuntimeError("The explicitly assigned GPU pools are not empty")
            self.reconfiguration_application, state = configuration_application(self)
            self._managed_rollout = state["managed_rollout"]
            layout = (
                state["config"]
                if state["pending"] is None
                else state["pending"]["source"]["config"]
            )
            target = ExecutionLayout.decode(layout)
            if target != self.execution_layout:
                self._set_execution_layout(target)
            self._event("STOPPED")
        except BaseException:
            self.consumer.close()
            raise

    def _restore_data_progress(self) -> None:
        """Recover logical counts independently of optional measurement history."""
        progress = (
            None
            if self.latest_manifest is None
            else self.latest_manifest.get("data_progress")
        )
        if progress is None:
            from areal.infra.rpc.serialization import deserialize_value

            progress = {"samples": 0, "effective_tokens": 0}
            for payload in self.consumer.applied_payloads():
                progress["samples"] += len(payload["episode_ids"])
                progress["effective_tokens"] += sum(
                    int(deserialize_value(value)["loss_mask"].sum())
                    for value in payload["trajectories"]
                )
        if (
            not isinstance(progress, dict)
            or any(
                type(progress.get(key)) is not int or progress[key] < 0
                for key in ("samples", "effective_tokens")
            )
            or progress["samples"] != self.applied_version * self.inbox.batch_size
            or self.applied_version > 0
            and progress["effective_tokens"] == 0
        ):
            raise ValueError(
                "Logical data progress disagrees with the checkpoint/queue cursor"
            )
        self.data_progress = dict(progress)
        self._active_data_progress = None

    @property
    def actor_world_size(self) -> int:
        return self.allocation.parallel.world_size

    @property
    def actor_backend(self) -> str:
        return self._actor_backend

    @property
    def published_version(self) -> int:
        """Latest weight version acknowledged by every serving replica."""
        return self.inbox.status()["published_version"]

    @property
    def checkpoint_version(self) -> int:
        """Latest durably saved full training state; zero denotes the pinned base."""
        return self.latest_manifest["global_step"] if self.latest_manifest else 0

    def control_snapshot(self) -> ControlSnapshot:
        with self._lifecycle_lock:
            published = self.published_version
            ready = (
                self.owner is None
                if self.actor is None
                else self.ready_since is not None
            )
            return ControlSnapshot(
                self._closed,
                not self._closed
                and ready
                and self.active is None
                and not self._publication_uncertain
                and not self._reconfiguration_incomplete
                and published == self.applied_version,
                self.generation,
                self.applied_version,
                published,
                json_digest(
                    {
                        "checkpoint": None
                        if self.latest_path is None
                        else self.latest_path.name,
                        "cursor": None
                        if self.latest_manifest is None
                        else self.latest_manifest["data_cursor"],
                        "queue_id": self.inbox.queue.queue_id,
                    }
                ),
                checkpoint_version=self.checkpoint_version,
            )

    @contextmanager
    def control_operation(self, *, blocking: bool = False):
        if not self._lifecycle_lock.acquire(blocking=blocking):
            raise ControlBusy("Native lifecycle is busy")
        try:
            yield self.control_snapshot()
        finally:
            self._lifecycle_lock.release()

    def fence_admission(self) -> None:
        self.inbox.set_admission_open(False)

    def validate_actor_dp(self, world_size: int) -> None:
        from areal.api.alloc_mode import ModelAllocation

        if type(world_size) is not int or world_size < 1:
            raise ValueError("Actor DP size must be a positive integer")
        if world_size == self.actor_world_size:
            return
        initial = ModelAllocation.from_str(self.config.actor.backend)
        if initial.backend != "fsdp" or any(
            getattr(initial.parallel, name) != 1
            for name in ("tp_size", "pp_size", "cp_size", "ep_size", "etp_size")
        ):
            raise ValueError("Actor reconfiguration supports pure FSDP DP only")
        if (
            world_size > len(self.resource_reservation)
            or self.inbox.batch_size % world_size
        ):
            raise ValueError("Actor DP must fit the reservation and divide the batch")
        if not self.reconfigurator_adapter.supports_actor_dp_resize:
            raise ValueError("Actor changes require a native Online adapter")

    def _set_actor_dp(self, world_size: int) -> None:
        from areal.api.alloc_mode import ModelAllocation

        initial = ModelAllocation.from_str(self.config.actor.backend)
        self._actor_backend = (
            self.config.actor.backend
            if world_size == initial.parallel.world_size
            else f"fsdp:d{world_size}"
        )
        self.allocation = ModelAllocation.from_str(self._actor_backend)

    @property
    def supports_resource_reconfiguration(self) -> bool:
        from areal.api.alloc_mode import ModelAllocation

        actor = ModelAllocation.from_str(self.config.actor.backend)
        rollout = ModelAllocation.from_str(self.config.rollout.backend)
        return (
            actor.backend == "fsdp"
            and all(
                getattr(actor.parallel, key) == 1
                for key in ("tp_size", "pp_size", "cp_size", "ep_size", "etp_size")
            )
            and rollout.backend == "sglang"
            and rollout.parallel.tp_size == rollout.parallel.pp_size == 1
        )

    @property
    def execution_layout(self) -> ExecutionLayout:
        return ExecutionLayout(
            self.actor_world_size, self.actor_devices, self.rollout_devices
        )

    @property
    def managed_rollout(self) -> bool:
        return self._managed_rollout or (
            self.rollout is not None
            and getattr(self.rollout, "resource_reconfiguration_enabled", False)
        )

    def validate_execution_layout(self, target: ExecutionLayout) -> None:
        source = self.execution_layout
        if target == source:
            return
        if not set(target.actor_devices + target.rollout_devices) <= set(
            self.resource_reservation
        ):
            raise ValueError("Resource target exceeds the original job reservation")
        self.validate_actor_dp(target.world_size)
        if not self.reconfigurator_adapter.supports_actor_dp_resize:
            raise ValueError("Reconfiguration requires a native Online adapter")
        pools_changed = (
            target.actor_devices != source.actor_devices
            or target.rollout_devices != source.rollout_devices
        )
        if pools_changed and (
            not self.supports_resource_reconfiguration or not self.managed_rollout
        ):
            raise ValueError(
                "GPU pool changes require managed local FSDP/SGLang TP1 resources"
            )
        if (
            target.actor_devices != source.actor_devices
            and self._actor_pool_setter is None
        ):
            raise ValueError("Actor pool changes require a local pool setter")
        if self.inbox.batch_size % target.world_size:
            raise ValueError("Actor DP count must divide the effective training batch")

    def _set_execution_layout(self, target: ExecutionLayout) -> None:
        self.validate_execution_layout(target)
        if target.actor_devices != self.actor_devices:
            self._actor_pool_setter(list(target.actor_devices))
        self.actor_devices, self.rollout_devices = (
            target.actor_devices,
            target.rollout_devices,
        )
        self.gpu_uuids = resolve_gpu_uuids(list(self.actor_devices))
        self._set_actor_dp(target.world_size)

    def plan_reconfiguration(self, target: ExecutionLayout) -> ReconfigurationPlan:
        """Compare layouts without changing engines, admission or durable state."""
        self.validate_execution_layout(target)
        return self.reconfiguration_planner.plan(
            self.execution_layout, target, actor_running=self.actor is not None
        )

    @_serialized
    @_timed_reconfiguration
    def reconfigure(
        self,
        *,
        target: ExecutionLayout,
        request_id: str,
        expected_generation: int,
        expected_weights_version: int,
        reconfigurator_bundle_id: str = "manual",
    ):
        return self.reconfiguration_application.executor.reconfigure(
            target=target,
            request_id=request_id,
            expected_generation=expected_generation,
            expected_weights_version=expected_weights_version,
            reconfigurator_bundle_id=reconfigurator_bundle_id,
        )

    def _reconfiguration_phase(self, name: str):
        return (
            nullcontext()
            if self._transition_timing is None
            else self._transition_timing.phase(name)
        )

    def _arm_post_reconfiguration(self) -> None:
        timing = self._transition_timing
        if timing is not None and callable(
            getattr(self.rollout, "arm_reconfiguration_observation", None)
        ):
            try:
                with timing.phase("arm_first_inference"):
                    self.rollout.arm_reconfiguration_observation(timing.transition_id)
                self._pending_transition_timing = timing
            except Exception:
                logger.warning("First-inference observation unavailable", exc_info=True)

    def _record_post_reconfiguration(self) -> None:
        timing = self._pending_transition_timing
        if timing is None:
            return
        try:
            rows = self.rollout.reconfiguration_metrics()
            timing.record("first_inference_snapshot", proxies=rows)
            self._pending_transition_timing = None
        except Exception:
            logger.warning("Cannot collect first-inference observation", exc_info=True)

    def _read_latest(self) -> dict[str, Any] | None:
        if not (self.store.root / "latest.json").exists():
            self.latest_path, self.latest_manifest = None, None
        else:
            self.latest_path, self.latest_manifest = self.store.latest(
                self.compatibility
            )
            layout_path = self.latest_path / "actor-layout.json"
            from areal.api.alloc_mode import ModelAllocation

            if layout_path.exists():
                layout = json.loads(layout_path.read_text())
                placement_valid = (
                    layout.get("reservation") == list(self.resource_reservation)
                    and set(layout["gpu_pool"]) <= set(self.resource_reservation)
                    and set(layout.get("rollout_devices", ()))
                    <= set(self.resource_reservation)
                    and not set(layout["gpu_pool"])
                    & set(layout.get("rollout_devices", ()))
                    if layout.get("format_version") == 2
                    else layout["gpu_pool"] == list(self.options.actor_devices)
                )
                if (
                    layout["world_size"] != self.latest_manifest["world_size"]
                    or not placement_valid
                ):
                    raise ValueError(
                        "Checkpoint Actor layout disagrees with its manifest/reservation"
                    )
                backend = layout["backend"]
            else:
                backend = self.config.actor.backend
            source = ModelAllocation.from_str(backend)
            if (
                source.backend != self.allocation.backend
                or source.parallel.world_size != self.latest_manifest["world_size"]
            ):
                raise ValueError("Checkpoint Actor layout is missing or inconsistent")
        return self.latest_manifest

    def _event(self, state: str, **details: Any) -> None:
        event = {
            "state": state,
            "wall_time": time.time(),
            "monotonic": time.monotonic(),
            "generation": self.generation,
            "applied_version": getattr(self, "applied_version", 0),
            "checkpoint_version": self.checkpoint_version,
            "owner": self.owner,
            "actor_world_size": self.actor_world_size,
            "actor_backend": self.actor_backend,
            "actor_gpu_pool": list(self.actor_devices),
            "rollout_gpu_pool": list(self.rollout_devices),
            **details,
        }
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(self.root / "state.json", event)

    def bind(
        self,
        rollout: Any,
        weight_meta: Any,
        *,
        capture_cpu: Callable[[], dict[str, Any]],
        restore_cpu: Callable[[dict[str, Any]], None],
    ) -> None:
        self.rollout, self.weight_meta = rollout, weight_meta
        self._capture_cpu, self._restore_cpu = capture_cpu, restore_cpu
        if self.latest_path is not None:
            restore_cpu(json.loads((self.latest_path / "coordinator.json").read_text()))
            self._restore_checkpoint_frequency()
        self._initial_cpu_state = deepcopy(capture_cpu())
        rollout._ensure_initialized()
        versions = rollout.serving_versions()
        rolled_back = self.inbox.status()["published_version"] > self.applied_version
        if rolled_back:
            rollout.pause_generation()
            self.inbox.rollback(self.consumer)
        if not versions or any(version != self.applied_version for version in versions):
            if self.latest_path is None and not rolled_back:
                raise RuntimeError(
                    "Initial serving weights do not match the immutable base version"
                )
            self.ensure_actor()
            self._publish()
            rollout.continue_generation()
        elif self.inbox.status()["published_version"] != self.applied_version:
            self.inbox.confirm_publication(self.applied_version, consumer=self.consumer)
        if rolled_back:
            rollout.continue_generation()
        self.reconfigurator_adapter.bind(self)
        self._revision_hooks_bound = self.revision_hooks is not None
        application = self.reconfiguration_application
        state = application.state()
        if "managed_rollout" not in state:
            atomic_json(
                application.root / "state.json",
                {**state, "managed_rollout": self.managed_rollout},
            )
        application.recover()
        self.inbox.set_admission_open(True)
        if self.managed_rollout:
            rollout.continue_generation()
            rollout.open_model_calls()

    def _restore_checkpoint_frequency(self) -> None:
        path = (
            self.latest_path / "checkpoint-schedule.json" if self.latest_path else None
        )
        self.checkpoint_frequency.load_state_dict(
            json.loads(path.read_text())
            if path is not None and path.exists()
            else deepcopy(self._initial_checkpoint_frequency)
        )

    def _assert_pool_free(self) -> None:
        present = gpu_processes()
        if any(present.get(gpu) for gpu in self.gpu_uuids):
            raise RuntimeError("Actor GPU pool is not confirmed empty")

    @_serialized
    def ensure_actor(self, *, connect_weights: bool = True) -> Any:
        if self.actor is not None:
            return self.actor
        if self.owner is not None:
            raise RuntimeError(
                "Previous Actor generation still requires release confirmation"
            )
        self._assert_pool_free()
        self.owner = uuid.uuid4().hex
        self.generation += 1
        self.started_at = time.monotonic()
        self._event("STARTING")
        actor_config = deepcopy(self.config.actor)
        actor_config.backend = self.actor_backend
        for spec in actor_config.scheduling_spec:
            spec.env_vars[OWNER_ENV] = self.owner
        self.last_restore_reports = self.last_restore_plan = None
        with self._reconfiguration_phase("actor_start"):
            self.actor = self.actor_factory(actor_config)
            self.actor.create_process_group(parallel_strategy=self.allocation.parallel)
            self.actor.initialize(
                role="actor",
                addr=None,
                ft_spec=self.ft_spec,
                wait=True,
                worker_seed=self.config.seed,
            )
        if self.latest_path is not None:
            restore_options = {}
            if self.latest_manifest["world_size"] != self.actor_world_size:
                from areal.v2.training_service.dp_restore import DPStateRestorePlan

                self.validate_actor_dp(self.actor_world_size)
                restore_options["restore_plan"] = dataclasses.asdict(
                    DPStateRestorePlan(
                        identity_sha256=self.identity,
                        checkpoint_generation=self.latest_path.name,
                        policy_version=self.applied_version,
                        source_world_size=self.latest_manifest["world_size"],
                        target_world_size=self.actor_world_size,
                        transition_id=json_digest(
                            {
                                "checkpoint": self.latest_path.name,
                                "target_dp": self.actor_world_size,
                                "run": self.identity,
                            }
                        ),
                    )
                )
            self.last_restore_plan = restore_options.get("restore_plan")
            with self._reconfiguration_phase("actor_restore"):
                reports = self.actor.full_checkpoint(
                    "load",
                    path=str(self.latest_path),
                    backend=self.allocation.backend,
                    identity_sha256=self.identity,
                    version=self.applied_version,
                    **restore_options,
                )
            self.last_restore_reports = reports
            if isinstance(reports, list):
                self._event(
                    "RESTORED",
                    worker_devices=[row.get("visible_devices") for row in reports],
                )
        if self.config.perf_tracer is not None:
            self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")
        if connect_weights:
            with self._reconfiguration_phase("awex_connect"):
                self.actor.connect_engine(self.rollout, self.weight_meta)
        self.ready_since = time.monotonic()
        self.idle_since = self.ready_since
        self._event("READY", startup_seconds=self.ready_since - self.started_at)
        self.monitor.record_phase(
            "restore" if self.latest_path else "start",
            self.ready_since - self.started_at,
        )
        return self.actor

    def next_batch(self) -> list[dict[str, Any]]:
        while True:
            with self._lifecycle_lock:
                if self._closed:
                    raise RuntimeError("On-demand coordinator is closed")
                batch = self._poll_batch()
            if batch is not None:
                return batch
            time.sleep(self.options.poll_interval)

    def _poll_batch(self) -> list[dict[str, Any]] | None:
        from areal.infra.rpc.serialization import deserialize_value

        if self.active is not None:
            raise RuntimeError("The previous native PPO update is still active")
        poll_reconfiguration_request(self)
        if self.reconfigurator_adapter.poll():
            return None
        if self._revision_hooks_bound:
            self.revision_hooks.poll()
        if self.inbox.status()["published_version"] != self.applied_version:
            self.ensure_actor()
            self._publish()
            self.rollout.continue_generation()
        identity = self.inbox.seal_batch(self.applied_version, consumer=self.consumer)
        if identity is not None:
            if self.reconfigurator_adapter.decide(self.consumer.peek()) != "train":
                return None
            self.ensure_actor()
            batch = self.consumer.claim()
            if (
                batch is None
                or batch.batch_id != identity
                or batch.policy_version != self.applied_version
            ):
                raise RuntimeError(
                    "The reserved batch does not match restored Actor progress"
                )
            trajectories = [
                deserialize_value(value) for value in batch.payload["trajectories"]
            ]
            for trajectory in trajectories:
                minimum, maximum = self.inbox._inspect_payload(trajectory)
                if (
                    maximum > self.applied_version
                    or self.applied_version - minimum > self.inbox.max_staleness
                ):
                    raise RuntimeError(
                        "Reserved token versions violate the native training bound"
                    )
            self._active_data_progress = {
                "samples": len(batch.payload["episode_ids"]),
                "effective_tokens": sum(
                    int(row["loss_mask"].sum()) for row in trajectories
                ),
            }
            self.monitor.begin_update(batch)
            self.active = batch
            self._update_applied = False
            self._train_started = time.monotonic()
            self.idle_since = None
            self._event(
                "TRAINING",
                batch_id=identity,
                episode_ids=batch.payload["episode_ids"],
            )
            return trajectories
        return None

    @_serialized
    def publish_update(self, version: int) -> None:
        """Mark the PPO update applied and publish it; checkpointing is independent."""
        if (
            self.actor is None
            or self._publication_uncertain
            or self.active is None
            or version != self.applied_version + 1
        ):
            raise RuntimeError("No matching native PPO update is ready to publish")
        self.monitor.record_phase("train", time.monotonic() - self._train_started)
        self.actor.set_version(version)
        self.consumer.mark_applied(self.active)
        self.data_progress = {
            key: self.data_progress[key] + self._active_data_progress[key]
            for key in ("samples", "effective_tokens")
        }
        self.applied_version = version
        self._update_applied = True
        self.rollout.pause()
        self._publish()

    @_serialized
    def maybe_checkpoint(self, *, epoch_step: int) -> bool:
        if self.config.recover.mode in ("disabled", "off"):
            return False
        if not self.checkpoint_frequency.check(
            epochs=int(epoch_step == self.ft_spec.steps_per_epoch - 1), steps=1
        ):
            return False
        return self.save_checkpoint(reason="periodic")

    @_serialized
    def save_checkpoint(self, *, reason: str) -> bool:
        """Persist full restart state at a save boundary or before planned release."""
        if self.applied_version == 0 or (
            self.latest_manifest is not None
            and self.latest_manifest["global_step"] == self.applied_version
            and not self._checkpoint_dirty
        ):
            return False
        if (
            self.actor is None
            or self._publication_uncertain
            or self.active is not None
            and not self._update_applied
            or self.inbox.status()["published_version"] != self.applied_version
        ):
            raise RuntimeError("Checkpoint requires applied, fully published weights")
        started = time.monotonic()

        def write(path: Path) -> None:
            self.last_checkpoint_reports = self.actor.full_checkpoint(
                "save",
                path=str(path),
                backend=self.allocation.backend,
                identity_sha256=self.identity,
                version=self.applied_version,
            )
            atomic_json(
                path / "actor-layout.json",
                {
                    "backend": self.actor_backend,
                    "world_size": self.actor_world_size,
                    "format_version": 2,
                    "gpu_pool": list(self.actor_devices),
                    "rollout_devices": list(self.rollout_devices),
                    "reservation": list(self.resource_reservation),
                },
            )
            atomic_json(path / "coordinator.json", self._capture_cpu())
            atomic_json(
                path / "checkpoint-schedule.json",
                self.checkpoint_frequency.state_dict(),
            )

        generation = self.store.commit(
            step=self.applied_version,
            world_size=self.allocation.parallel.world_size,
            compatibility=self.compatibility,
            write=write,
            estimated_bytes=self.options.checkpoint_estimated_bytes,
            data_cursor=self.consumer.cursor,
            data_progress=self.data_progress,
        )
        manifest = self._read_latest()
        if self.latest_path != generation or manifest is None:
            raise RuntimeError("Checkpoint publication requires reconciliation")
        self.consumer.acknowledge(manifest)
        self._checkpoint_dirty = False
        # Optional observations are written after the strict full-state commit;
        # their durability must never decide whether model/cursor saving succeeds.
        self.monitor.checkpoint_state(generation)
        self._event(
            "CHECKPOINTED",
            reason=reason,
            checkpoint=generation.name,
            checkpoint_seconds=time.monotonic() - started,
        )
        self.monitor.record_phase("checkpoint", time.monotonic() - started)
        return True

    @_serialized
    def finish_update(self) -> None:
        """Resume serving only after saving, evaluation, cleanup and logging finish."""
        if (
            self.active is None
            or not self._update_applied
            or self.inbox.status()["published_version"] != self.applied_version
        ):
            raise RuntimeError("No fully published native PPO update to finish")
        self.rollout.continue_generation()
        self.rollout.resume()
        self.active = None
        self._update_applied = False
        # Evaluation/logging may change CPU or RNG state after a periodic save.
        # A planned release must snapshot the completed boundary as well.
        self._checkpoint_dirty = True
        self.idle_since = time.monotonic()
        self.monitor.complete_update()
        self._record_post_reconfiguration()
        self._event("READY", serving_versions=self.rollout.serving_versions())

    def _publish(self) -> None:
        if self.actor is None or self.rollout is None:
            raise RuntimeError("Publication requires live native Actor and Rollout")
        started = time.monotonic()
        self._event("PUBLISHING")
        self._publication_uncertain = True
        self.actor.update_weights(
            self.weight_meta.with_version(self.applied_version), resume_generation=False
        )
        versions = self.rollout.serving_versions()
        if not versions or any(version != self.applied_version for version in versions):
            raise RuntimeError(
                "Serving replicas have not all acknowledged updated weights"
            )
        self.inbox.confirm_publication(self.applied_version, consumer=self.consumer)
        self._publication_uncertain = False
        self.monitor.record_phase("publish", time.monotonic() - started)
        self._event(
            "PUBLISHED",
            publication_seconds=time.monotonic() - started,
            serving_versions=versions,
        )

    def _release(self, *, training_uncertain: bool = False) -> None:
        training_uncertain = training_uncertain or self._publication_uncertain
        if self.actor is None and self.owner is None:
            return
        release_started = time.monotonic()
        self._cleanup_event("RELEASING")
        if self.actor is not None:
            # No resource creator may outlive the ownership probe below.
            self.actor.quiesce_initialization()
            if (
                not training_uncertain
                and self.ready_since is not None
                and self.config.perf_tracer is not None
                and self.config.perf_tracer.enabled
            ):
                try:
                    self.actor.save_perf_tracer(force=True)
                except Exception:
                    logger.warning(
                        "Actor tracer flush failed during release", exc_info=True
                    )
                    training_uncertain = True
            try:
                if training_uncertain:
                    # A timed-out collective still owns its worker thread. Do
                    # not queue tracer or teardown RPCs behind that collective.
                    if self.owner is None:
                        raise RuntimeError("Uncertain Actor has no process owner")
                    stop_owned_processes(self.owner)
                    self._confirm_stopped()
                    self.actor.destroy(training_stopped=True)
                else:
                    self.actor.destroy()
            except Exception:
                if training_uncertain:
                    raise
                # A dead Actor cannot acknowledge AWEX teardown. Only after
                # proving its entire generation has stopped may the surviving
                # inference peers be disconnected without training-side RPCs.
                if self.owner is None:
                    raise
                stop_owned_processes(self.owner)
                self._confirm_stopped()
                self.actor.destroy(training_stopped=True)
        if self.owner is not None:
            stop_owned_processes(self.owner)
        self._confirm_stopped()
        gpu_seconds = (time.monotonic() - self.started_at) * self.actor_world_size
        self.actor, self.owner = None, None
        self.ready_since, self.idle_since, self.started_at = None, None, None
        self._cleanup_event("STOPPED", actor_gpu_seconds=gpu_seconds)
        self.monitor.released(time.monotonic() - release_started, gpu_seconds)

    def _confirm_stopped(self) -> None:
        if self.owner is not None and owned_processes(self.owner):
            raise RuntimeError("Actor descendants remain alive")
        self._assert_pool_free()

    def _cleanup_event(self, state: str, **details: Any) -> None:
        try:
            self._event(state, **details)
        except Exception:
            logger.error("Cannot persist lifecycle cleanup evidence", exc_info=True)

    @_serialized
    def recover_uncertain(self, error: Exception) -> None:
        """Roll back a failed generation to its latest full checkpoint and replay data."""
        try:
            self.inbox.set_admission_open(False)
        except Exception:
            logger.error("Cannot persist the recovery admission fence", exc_info=True)
        self._cleanup_event("RECOVERING", error=str(error))
        self.rollout.pause()
        # The inference scheduler may be waiting in the failed AWEX collective
        # too. Fence local submission, then require inference teardown to finish
        # after the failed training generation is proved stopped.
        self._release(training_uncertain=True)
        self.rollout.pause_generation()
        manifest = self._read_latest()
        self.consumer.restore_checkpoint(manifest)
        self.inbox.rollback(self.consumer)
        self.applied_version = manifest["global_step"] if manifest else 0
        self._restore_data_progress()
        self.active = None
        self._update_applied = False
        self._checkpoint_dirty = False
        self._restore_checkpoint_frequency()
        if self.latest_path is not None:
            self._restore_cpu(
                json.loads((self.latest_path / "coordinator.json").read_text())
            )
        else:
            self._restore_cpu(deepcopy(self._initial_cpu_state))
        application = self.reconfiguration_application
        journal = application.root / "state.json"
        if journal.exists():
            state = application.state()
            layout = (
                state["config"]
                if state["pending"] is None
                else state["pending"]["source"]["config"]
            )
            target = ExecutionLayout.decode(layout)
            restored_layout = (
                target != self.execution_layout or self._reconfiguration_incomplete
            )
            if restored_layout:
                application.executor.recover_layout(target)
            else:
                self.monitor.recover()
        else:
            restored_layout = False
            self.monitor.recover()
        if self._revision_hooks_bound:
            self.revision_hooks.recover()
        # Failed transfers may have overwritten only some inference tensors
        # without advancing any version label. A matching label is not proof
        # that serving weights equal the checkpoint. Restore every tensor and
        # require publication acknowledgment before reopening generation.
        if not restored_layout:
            self.ensure_actor()
            self._publish()
        self._reconfiguration_incomplete = False
        self.rollout.continue_generation()
        self.rollout.resume()
        self.monitor.complete_update()
        if journal.exists():
            application.recover()
            self.rollout.continue_generation()
            self.rollout.resume()
        self.inbox.set_admission_open(True)
        if self.managed_rollout:
            self.rollout.open_model_calls()

    @_serialized
    def close(self) -> None:
        if self._closed:
            return
        errors = []
        self._record_post_reconfiguration()
        try:
            self.inbox.set_admission_open(False)
        except Exception as error:
            errors.append(error)
        try:
            if self.active is None and self.actor is not None:
                self.save_checkpoint(reason="close")
        except Exception as error:
            if (
                self.latest_manifest is None
                or self.latest_manifest["global_step"] != self.applied_version
                or self._checkpoint_dirty
            ):
                raise ExceptionGroup(
                    "Checkpoint incomplete; retain Actor and consumer ownership for retry",
                    [*errors, error],
                ) from error
            # Phase/journal reporting can fail after the full snapshot committed.
            # Preserve that error while still cleaning up the confirmed state.
            errors.append(error)
        try:
            self._release(training_uncertain=self.active is not None)
        except Exception as error:
            if self.actor is not None or self.owner is not None:
                raise ExceptionGroup(
                    "Actor release incomplete; retain consumer ownership",
                    [*errors, error],
                ) from error
            # Measurement persistence can fail after release was proved. It must
            # not keep the queue fence or revision worker alive.
            errors.append(error)
        try:
            if self.reconfigurator_adapter.owner in (None, self):
                self.reconfigurator_adapter.close()
        except Exception as error:
            errors.append(error)
        finally:
            self.consumer.close()
            self._closed = True
        if errors:
            raise ExceptionGroup("Coordinator cleanup incomplete", errors)
