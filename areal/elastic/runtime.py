# SPDX-License-Identifier: Apache-2.0

"""CPU-owned lifecycle around the native PPO Actor and persistent inline batch source."""

from __future__ import annotations

import dataclasses
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any

from areal.elastic.control.processes import (
    OWNER_ENV,
    gpu_processes,
    owned_processes,
    resolve_gpu_uuids,
    stop_owned_processes,
)
from areal.elastic.policy import LifecyclePolicy, validate_policy
from areal.elastic.state.batch_queue import QueuedBatch
from areal.elastic.state.checkpoint_store import CheckpointStore
from areal.elastic.state.inline_source import InlineBatchSource
from areal.utils import logging
from areal.utils.atomic import (
    artifact_manifest,
    atomic_json,
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


def validate_on_demand(config: Any) -> None:
    """Reject unsupported state/placement combinations before starting services."""
    from areal.api.alloc_mode import ModelAllocation
    from areal.utils.environ import is_single_controller

    lifecycle = config.actor_lifecycle
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
    if config.rollout.agent is None or config.rollout.agent.mode != "inline":
        raise ValueError("on_demand requires agent.mode='inline'")
    if lifecycle.max_recovery_attempts != 0:
        raise ValueError(
            "Inline demo supports planned Actor restarts, not coordinator recovery"
        )
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
    if (actor.parallel.world_size, rollout.parallel.world_size) != (
        len(lifecycle.actor_devices),
        len(lifecycle.rollout_devices),
    ):
        raise ValueError(
            "Each fixed GPU pool must exactly match its native parallel layout"
        )
    if (
        config.gconfig.n_samples < 1
        or config.gconfig.reward_normalization
        or config.dynamic_bs
    ):
        raise ValueError(
            "on_demand requires positive n_samples, fixed batches and training-time reward normalization"
        )
    if config.gconfig.n_samples > 1 and (
        config.rollout.agent.export_style != "concat"
        or not config.rollout.agent.drop_retry_orphans
    ):
        raise ValueError("Grouped on_demand requires concat with drop_retry_orphans")
    if config.rollout.consumer_batch_size != config.train_dataset.batch_size:
        raise ValueError(
            "Rollout consumer_batch_size must equal the dataset task-group batch size"
        )
    for norm in (config.actor.adv_norm, config.actor.reward_norm):
        if norm is not None and (
            norm.mean_level == "group" or norm.std_level == "group"
        ):
            if (
                config.gconfig.n_samples == 1
                or norm.group_size != config.gconfig.n_samples
            ):
                raise ValueError("Group normalization must match n_samples > 1")
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
        policy_adapter: LifecyclePolicy | None = None,
    ) -> None:
        validate_policy(policy_adapter)
        self.config, self.ft_spec, self.actor_factory = config, ft_spec, actor_factory
        self._lifecycle_lock = threading.RLock()
        self.policy_adapter = (
            policy_adapter
            if policy_adapter is not None
            else LifecyclePolicy(mode=config.actor_lifecycle.policy)
        )
        self._last_durations: dict[str, float] = {}
        self._train_started: float | None = None
        self.options = config.actor_lifecycle
        self.checkpoint_frequency = EpochStepTimeFreqCtl(
            freq_epoch=config.recover.freq_epochs,
            freq_step=config.recover.freq_steps,
            freq_sec=config.recover.freq_secs,
        )
        self._initial_checkpoint_frequency = deepcopy(
            self.checkpoint_frequency.state_dict()
        )
        self.root = Path(self.options.state_root)
        if self.root.exists() and any(self.root.iterdir()):
            raise ValueError(
                "Inline demo requires a fresh state directory; coordinator recovery is unsupported"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.inbox = InlineBatchSource(
            self.root / "inbox",
            batch_size=config.train_dataset.batch_size,
            max_staleness=config.rollout.max_head_offpolicyness,
            group_size=config.gconfig.n_samples,
        )
        from areal.api.alloc_mode import ModelAllocation

        self.allocation = ModelAllocation.from_str(config.actor.backend)
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
            "group_size": self.inbox.group_size,
            "export_style": config.rollout.agent.export_style,
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
        self.cancelled = threading.Event()
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
            if previous.exists():
                journal = json.loads(previous.read_text())
                self.generation = journal["generation"]
                if journal.get("owner"):
                    raise RuntimeError(
                        "Previous coordinator left an unconfirmed Actor generation; preserve its journal for cleanup"
                    )
            self.gpu_uuids = resolve_gpu_uuids(list(self.options.actor_devices))
            rollout_gpus = resolve_gpu_uuids(list(self.options.rollout_devices))
            present = gpu_processes()
            if any(present.get(gpu) for gpu in self.gpu_uuids + rollout_gpus):
                raise RuntimeError("The explicitly assigned GPU pools are not empty")
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

            progress = {"task_groups": 0, "samples": 0, "effective_tokens": 0}
            for payload in self.consumer.applied_payloads():
                progress["task_groups"] += len(payload["episode_ids"])
                for value in payload["trajectories"]:
                    trajectory = deserialize_value(value)
                    progress["samples"] += trajectory["input_ids"].shape[0]
                    progress["effective_tokens"] += int(trajectory["loss_mask"].sum())
        if (
            not isinstance(progress, dict)
            or any(
                type(progress.get(key)) is not int or progress[key] < 0
                for key in ("task_groups", "samples", "effective_tokens")
            )
            or progress["task_groups"] != self.applied_version * self.inbox.batch_size
            or progress["samples"] < progress["task_groups"]
            or self.inbox.group_size > 1
            and progress["samples"] != progress["task_groups"] * self.inbox.group_size
            or self.applied_version > 0
            and progress["effective_tokens"] == 0
        ):
            raise ValueError(
                "Logical data progress disagrees with the checkpoint/queue cursor"
            )
        self.data_progress = dict(progress)
        self._active_data_progress = None

    @property
    def published_version(self) -> int:
        """Latest weight version acknowledged by every serving replica."""
        return self.inbox.status()["published_version"]

    @property
    def checkpoint_version(self) -> int:
        """Latest durably saved full training state; zero denotes the pinned base."""
        return self.latest_manifest["global_step"] if self.latest_manifest else 0

    def _read_latest(self) -> dict[str, Any] | None:
        if not (self.store.root / "latest.json").exists():
            self.latest_path, self.latest_manifest = None, None
        else:
            self.latest_path, self.latest_manifest = self.store.latest(
                self.compatibility
            )
        return self.latest_manifest

    def _event(self, state: str, **details: Any) -> None:
        for key in (
            "startup_seconds",
            "checkpoint_seconds",
            "update_seconds",
            "release_seconds",
        ):
            if key in details:
                self._last_durations[key] = details[key]
        status = self.get_status()
        event = {
            "rollout": status["rollout"],
            "counts": status["counts"],
            "state": state,
            "wall_time": time.time(),
            "monotonic": time.monotonic(),
            "generation": self.generation,
            "applied_version": getattr(self, "applied_version", 0),
            "checkpoint_version": self.checkpoint_version,
            "actor_devices": list(self.options.actor_devices),
            "rollout_devices": list(self.options.rollout_devices),
            "owner": self.owner,
            **details,
        }
        with (self.root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(self.root / "state.json", event)
        atomic_json(self.root / "status.json", self.get_status())

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
        self.policy_adapter.bind(self)
        self.inbox.set_admission_open(True)

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
    def ensure_actor(self) -> Any:
        if self.actor is not None:
            if self.ready_since is None:
                raise RuntimeError(
                    "Actor initialization/restore is incomplete; release it first"
                )
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
        for spec in actor_config.scheduling_spec:
            spec.env_vars[OWNER_ENV] = self.owner
        self.actor = self.actor_factory(actor_config)
        self.actor.create_process_group(parallel_strategy=self.allocation.parallel)
        self.actor.initialize(
            role="actor",
            addr=None,
            ft_spec=self.ft_spec,
            wait=True,
            worker_seed=self.config.seed,
            **(
                {"restore_from_full_checkpoint": True}
                if self.latest_path is not None
                and self.allocation.backend == "megatron"
                and self.config.actor.dtype == "bfloat16"
                and self.config.actor.megatron.fp8_config is None
                else {}
            ),
        )
        if self.latest_path is not None:
            reports = self.actor.full_checkpoint(
                "load",
                path=str(self.latest_path),
                backend=self.allocation.backend,
                identity_sha256=self.identity,
                version=self.applied_version,
            )
            self._event("RESTORED", reports=reports, checkpoint=str(self.latest_path))
        if self.config.perf_tracer is not None:
            self.actor.config_perf_tracer(self.config.perf_tracer, role="actor")
        self.actor.connect_engine(self.rollout, self.weight_meta)
        self.ready_since = time.monotonic()
        self.idle_since = self.ready_since
        self._event("READY", startup_seconds=self.ready_since - self.started_at)
        return self.actor

    def get_status(self, *, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        inbox = self.inbox.status()
        return {
            "actor_running": self.actor is not None,
            "coordinator_closed": self._closed,
            "last_measured_durations": dict(self._last_durations),
            "training_active": self.active is not None,
            "generation": self.generation,
            "applied_version": self.applied_version,
            "published_version": inbox["published_version"],
            "checkpoint_version": self.checkpoint_version,
            "batch_size": self.inbox.batch_size,
            "count_unit": inbox["count_unit"],
            "group_size": self.inbox.group_size,
            "trajectories_per_batch": self.inbox.batch_size * self.inbox.group_size,
            "counts": inbox["counts"],
            "rollout": inbox.get("rollout", {}),
            "source_running": inbox["source_running"],
            "idle_seconds": 0.0
            if self.idle_since is None
            else max(0.0, now - self.idle_since),
            "residence_seconds": 0.0
            if self.ready_since is None
            else max(0.0, now - self.ready_since),
            "actor_devices": list(self.options.actor_devices),
            "rollout_devices": list(self.options.rollout_devices),
        }

    @_serialized
    def start_actor(self) -> dict[str, Any]:
        """Start/restore the fixed worker group at the coordinator boundary."""
        if self.active is not None or self._publication_uncertain:
            raise RuntimeError("Actor start requires an idle, published boundary")
        self.ensure_actor()
        return self.get_status()

    @_serialized
    def stop_actor(self) -> dict[str, Any]:
        """Save and release only at a completed boundary with no ready batch."""
        if self.actor is None:
            return self.get_status()
        if self.active is not None or self._publication_uncertain:
            raise RuntimeError("Actor release requires a completed update")
        if self.published_version != self.applied_version:
            raise RuntimeError("Actor release requires confirmed publication")
        if (
            self.inbox.seal_batch(self.applied_version, consumer=self.consumer)
            is not None
        ):
            raise RuntimeError("Actor release refused: a complete batch is ready")
        self.save_checkpoint(reason="release")
        if (
            self.inbox.seal_batch(self.applied_version, consumer=self.consumer)
            is not None
        ):
            raise RuntimeError("A batch arrived during checkpoint; retain Actor")
        self._release()
        return self.get_status()

    def next_batch(self) -> list[dict[str, Any]]:
        while True:
            with self._lifecycle_lock:
                if self._closed or self.cancelled.is_set():
                    raise RuntimeError("On-demand coordinator is closed or cancelled")
                batch = self._poll_batch()
            if batch is not None:
                return batch
            time.sleep(self.options.poll_interval)

    def _poll_batch(self) -> list[dict[str, Any]] | None:
        from areal.infra.rpc.serialization import deserialize_value

        self.policy_adapter.poll_commands()
        atomic_json(self.root / "status.json", self.get_status())
        if self.active is not None:
            raise RuntimeError("The previous native PPO update is still active")
        if self.inbox.status()["published_version"] != self.applied_version:
            self.ensure_actor()
            self._publish()
            self.rollout.continue_generation()
        identity = self.inbox.seal_batch(self.applied_version, consumer=self.consumer)
        if identity is not None:
            if self.policy_adapter.decide(self.consumer.peek()) != "train":
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
            self.active = batch
            self._update_applied = False
            trajectories = [
                deserialize_value(value) for value in batch.payload["trajectories"]
            ]
            token_count = age_total = old_count = max_age = 0
            for trajectory in trajectories:
                minimum, maximum = self.inbox._inspect_payload(trajectory)
                self.inbox.inspect_group(trajectory)
                if maximum > self.applied_version:
                    raise RuntimeError(
                        "Reserved token version is ahead of the published Actor"
                    )
                # Native max_head_offpolicyness limits dispatch credits, not the
                # age of every token in out-of-order, multi-turn episodes. Keep
                # their original versions and native PPO importance weighting.
                mask = (
                    trajectory["loss_mask"].bool() & trajectory["attention_mask"].bool()
                )
                mask[:, 0] = False
                ages = self.applied_version - trajectory["versions"][mask]
                token_count += ages.numel()
                age_total += int(ages.sum())
                old_count += int((ages > self.inbox.max_staleness).sum())
                max_age = max(max_age, self.applied_version - minimum)
            self._active_data_progress = {
                "task_groups": len(batch.payload["episode_ids"]),
                "samples": sum(row["input_ids"].shape[0] for row in trajectories),
                "effective_tokens": sum(
                    int(row["loss_mask"].sum()) for row in trajectories
                ),
            }
            self._train_started = time.monotonic()
            self.idle_since = None
            self._event(
                "TRAINING",
                batch_id=identity,
                episode_ids=batch.payload["episode_ids"],
                traj_group_sizes=[row["input_ids"].shape[0] for row in trajectories],
                group_rewards=[row["rewards"].tolist() for row in trajectories],
                token_age_max=max_age,
                token_age_mean=age_total / token_count,
                tokens_older_than_head_window=old_count,
                effective_tokens=token_count,
                head_window=self.inbox.max_staleness,
            )
            return trajectories
        self.maybe_suspend()
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
        self.actor.set_version(version)
        self.consumer.mark_applied(self.active)
        self.data_progress = {
            key: self.data_progress[key] + self._active_data_progress[key]
            for key in ("task_groups", "samples", "effective_tokens")
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
            self.actor.full_checkpoint(
                "save",
                path=str(path),
                backend=self.allocation.backend,
                identity_sha256=self.identity,
                version=self.applied_version,
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
        self._event(
            "CHECKPOINTED",
            reason=reason,
            checkpoint=generation.name,
            checkpoint_seconds=time.monotonic() - started,
        )
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
        self._event(
            "READY",
            update_seconds=time.monotonic() - self._train_started,
            serving_versions=self.rollout.serving_versions(),
        )

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
        self._event(
            "PUBLISHED",
            publication_seconds=time.monotonic() - started,
            serving_versions=versions,
        )

    @_serialized
    def maybe_suspend(self, now: float | None = None) -> bool:
        if self.actor is None or self.active is not None or self._publication_uncertain:
            return False
        now = time.monotonic() if now is None else now
        if self.inbox.status()["published_version"] != self.applied_version:
            return False
        if (
            self.inbox.seal_batch(self.applied_version, consumer=self.consumer)
            is not None
        ):
            self.idle_since = None
            return False
        if self.idle_since is None:
            self.idle_since = now
        if now - self.ready_since < self.options.min_ready_seconds:
            return False
        if self.policy_adapter.decide(None, now=now) != "release":
            return False
        # Rollout continues while the control LLM decides. Recheck before the
        # expensive checkpoint, as well as after it, using the live queue.
        if (
            self.inbox.seal_batch(self.applied_version, consumer=self.consumer)
            is not None
        ):
            self.idle_since = None
            self._event("RELEASE_DEFERRED", reason="batch_ready_after_decision")
            return False
        self.save_checkpoint(reason="release")
        # Receipt threads can complete new episodes while the checkpoint is written.
        if (
            self.inbox.seal_batch(self.applied_version, consumer=self.consumer)
            is not None
        ):
            self.idle_since = None
            self._event("RELEASE_DEFERRED", reason="batch_ready_after_checkpoint")
            return False
        self._release()
        return True

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
                    self.actor.destroy(parallel_child_cleanup=True)
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
        gpu_seconds = (time.monotonic() - self.started_at) * len(self.gpu_uuids)
        self.actor, self.owner = None, None
        self.ready_since, self.idle_since, self.started_at = None, None, None
        self._cleanup_event(
            "STOPPED",
            actor_gpu_seconds=gpu_seconds,
            release_seconds=time.monotonic() - release_started,
        )

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
        # Failed transfers may have overwritten only some inference tensors
        # without advancing any version label. A matching label is not proof
        # that serving weights equal the checkpoint. Restore every tensor and
        # require publication acknowledgment before reopening generation.
        self.ensure_actor()
        self._publish()
        self.rollout.continue_generation()
        self.rollout.resume()
        self.inbox.set_admission_open(True)

    @_serialized
    def close(self) -> None:
        if self._closed:
            return
        errors = []
        try:
            self.inbox.close()
        except Exception as error:
            errors.append(error)
        try:
            self.inbox.set_admission_open(False)
        except Exception as error:
            errors.append(error)
        try:
            if (
                self.active is None
                and self.actor is not None
                and self.ready_since is not None
            ):
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
            self._release(
                training_uncertain=self.active is not None
                or (self.actor is not None and self.ready_since is None)
            )
        except Exception as error:
            if self.actor is not None or self.owner is not None:
                raise ExceptionGroup(
                    "Actor release incomplete; retain consumer ownership",
                    [*errors, error],
                ) from error
            # Measurement persistence can fail after release was proved. It must
            # not keep the queue fence alive.
            errors.append(error)
        try:
            if self.policy_adapter.owner in (None, self):
                self.policy_adapter.close()
        except Exception as error:
            errors.append(error)
        finally:
            self.consumer.close()
            self._closed = True
            atomic_json(self.root / "status.json", self.get_status())
        if errors:
            raise ExceptionGroup("Coordinator cleanup incomplete", errors)
