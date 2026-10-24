# SPDX-License-Identifier: Apache-2.0

"""Default CPU observations, causal phase history and recovery for Elastic."""

from __future__ import annotations

import dataclasses
import json
import time
from functools import wraps
from typing import Any

from areal.utils import logging
from areal.utils.atomic import atomic_json, json_digest

from ..contracts import (
    BatchWorkload,
    ElasticObservation,
    HistoryWindow,
    Measurement,
    TrainingConfig,
)
from ..state.batch_queue import QueuedBatch
from .history import RuntimeHistory

SCOPE = "native-async-v1"
logger = logging.getLogger("RLTrainer")


def _optional_metrics(method):
    """Only observation failures degrade; native execution invariants stay strict."""

    @wraps(method)
    def collect(self, *args, **kwargs):
        if not self.metrics_available:
            return None
        try:
            return method(self, *args, **kwargs)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.mark_unavailable(error)
            return None

    return collect


class ElasticMonitor:
    """Collect execution facts independently of policy selection and revision.

    The coordinator and inbox remain authoritative for training progress. Only
    completed, published updates enter the policy/revision history window.
    """

    def __init__(self, owner: Any):
        self.owner = owner
        self.runtime_history: RuntimeHistory | None = None
        self.history = HistoryWindow()
        self.completed_tokens = 0
        self.metrics_available = True
        self.last_error: str | None = None
        self.measurement_id: str | None = None
        self._recovery_context = None
        self._series_id = None
        self.pending = None
        self.phases = []
        self._recovering = False
        self._recovery_phases = []
        self._decision = None
        self._workload_cache = None
        self._history_sequence = 0

    @property
    def actor_world_size(self) -> int:
        return getattr(self.owner, "actor_world_size", len(self.owner.gpu_uuids))

    def bind(self, execution_id: str) -> None:
        if self.measurement_id is not None:
            raise RuntimeError("Monitor is already bound")
        self.measurement_id = execution_id
        self._recovery_context = json_digest(
            {
                "recovery_id": self.owner.identity,
                "queue_id": self.owner.inbox.queue.queue_id,
                "checkpoint_root": str(self.owner.store.root),
            }
        )
        # Startup reconciliation can precede binding; retain actual timings as
        # lifecycle boundaries rather than attach them to an old PPO update.
        startup = self.phases
        self.recover()
        if self.metrics_available:
            self._recovery_phases.extend(startup)
        self.complete_update()

    def mark_unavailable(self, error: Exception) -> None:
        """Disable measured decisions/revision until recovery reopens the series."""
        if self.metrics_available:
            logger.warning(
                "Elastic metrics unavailable; use Basic timeout and pause revision: %s",
                error,
            )
        self.metrics_available = False
        self.last_error = f"{type(error).__name__}: {error}"
        self.history = HistoryWindow()
        self.pending = self._decision = self._workload_cache = None
        self.phases = []
        self._recovery_phases = []
        self._recovering = False

    @property
    def pending_path(self):
        return self.runtime_history.path.with_name("monitor-pending.json")

    def _next_sequence(self):
        self._history_sequence += 1
        return self._history_sequence

    def _finish_recovery(self):
        if self._recovery_phases:
            self.runtime_history.append_boundary(
                {
                    "observation_sequence": self._next_sequence(),
                    "weights_version": self.owner.applied_version,
                    "checkpoint_id": None
                    if self.owner.latest_path is None
                    else self.owner.latest_path.name,
                    "actor_generation": self.owner.generation,
                    "recovery": True,
                    "measurements": [
                        {
                            "phase": row["phase"],
                            "seconds": row["seconds"],
                            "config": {"world_size": row["world_size"]},
                            "workload": None,
                        }
                        for row in self._recovery_phases
                    ],
                }
            )
            for row in self._recovery_phases:
                self.history.add_measurement(Measurement(**row))
        self._recovery_phases = []
        self._recovering = False

    def observe(
        self, batch: QueuedBatch | None = None, *, now: float | None = None
    ) -> ElasticObservation:
        """Map live progress into the shared planning schema.

        Its legacy ``committed`` flag means a completed, published update;
        ``committed_samples`` counts applied work, not checkpoint durability.
        """
        now = time.monotonic() if now is None else now
        state = self.owner.control_snapshot()
        return ElasticObservation(
            state="ACTIVE" if self.owner.actor is not None else "SUSPENDED",
            model_version=state.applied_version,
            serving_version=state.published_version,
            training_active=self.owner.actor is not None,
            batch_ready=batch is not None,
            next_batch=None if batch is None else self.workload(batch),
            idle_duration=0
            if self.owner.idle_since is None
            else max(0, now - self.owner.idle_since),
            gate_available=state.boundary_ready,
            pending_publication=state.published_version != state.applied_version,
            committed=self.owner.active is None
            and state.published_version == state.applied_version,
            observed_at=now,
            training_config=TrainingConfig(self.actor_world_size),
            committed_samples=self.owner.data_progress["samples"],
            committed_tokens=self.completed_tokens,
            max_train_gpus=len(self.owner.gpu_uuids),
            has_checkpoint=self.owner.latest_path is not None,
        )

    def workload(self, batch):
        if self._workload_cache is None or self._workload_cache[0] != batch.batch_id:
            self._workload_cache = (batch.batch_id, self._decode_workload(batch))
        return self._workload_cache[1]

    @staticmethod
    def _decode_workload(batch):
        from areal.infra.rpc.serialization import deserialize_value

        payloads = [deserialize_value(value) for value in batch.payload["trajectories"]]
        return BatchWorkload(
            len(batch.payload["episode_ids"]),
            sum(int(value["loss_mask"].sum()) for value in payloads),
            sum(int(value["attention_mask"].sum()) for value in payloads),
        )

    @_optional_metrics
    def record_decision(self, batch, observation, decision, action, *, policy: dict):
        if action == "train":
            self._decision = {
                "batch_id": batch.batch_id,
                "observation": dataclasses.asdict(observation),
                "history": dataclasses.asdict(self.history.snapshot()),
                "action": dataclasses.asdict(decision)
                if dataclasses.is_dataclass(decision)
                else decision,
                "policy": policy,
            }

    @_optional_metrics
    def begin_update(self, batch):
        if self._decision is None or self._decision["batch_id"] != batch.batch_id:
            raise ValueError("Monitor lacks the original resource decision")
        self.pending = {
            "batch_id": batch.batch_id,
            "target_version": self.owner.applied_version + 1,
            "observation": self._decision["observation"],
            "workload": self._decision["observation"]["next_batch"],
            "history": self._decision["history"],
            "decision": self._decision["action"],
            **self._decision["policy"],
        }
        self.pending["committed_tokens"] = (
            self.completed_tokens + self.pending["workload"]["effective_tokens"]
        )

    @_optional_metrics
    def record_phase(self, phase, seconds):
        """Record native conditional service time at existing execution boundaries."""
        measurement = Measurement(
            phase,
            self.actor_world_size,
            seconds,
            0 if self.pending is None else self.pending["workload"]["sequence_tokens"],
        )
        if self.runtime_history is None:
            self.phases.append(dataclasses.asdict(measurement))
            return
        if self._recovering:
            self._recovery_phases.append(dataclasses.asdict(measurement))
            return
        if phase == "checkpoint" and self.pending is None:
            rows = self.runtime_history.snapshot(self.owner.applied_version)["updates"]
            self.runtime_history.append_boundary(
                {
                    "observation_sequence": self._next_sequence(),
                    "weights_version": self.owner.applied_version,
                    "checkpoint_id": self.owner.latest_path.name,
                    "actor_generation": self.owner.generation,
                    "measurements": [
                        {
                            "phase": phase,
                            "seconds": seconds,
                            "config": {"world_size": self.actor_world_size},
                            "workload": rows[-1]["workload"] if rows else None,
                        }
                    ],
                }
            )
            self.history.add_measurement(
                dataclasses.replace(
                    measurement,
                    sequence_tokens=rows[-1]["workload"]["sequence_tokens"]
                    if rows
                    else 0,
                )
            )
            return
        self.phases.append(dataclasses.asdict(measurement))
        if phase in {"checkpoint", "publish"} and self.pending is not None:
            atomic_json(
                self.pending_path,
                {
                    "checkpoint_id": None
                    if self.owner.latest_path is None
                    else self.owner.latest_path.name,
                    "weights_version": self.pending["target_version"],
                    "format_version": 2,
                    "recovery_context": self._recovery_context,
                    "measurement_id": self.measurement_id,
                    "series_id": self._series_id,
                    "data_cursor": self.owner.consumer.cursor,
                    "context": self.runtime_history.context,
                    "pending": self.pending,
                    "measurements": self.phases,
                },
            )

    def checkpoint_state(self, path):
        """Optional metric snapshot; never a condition for saving training state."""
        if not self.metrics_available:
            return
        try:
            self._checkpoint_state(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.mark_unavailable(error)
            # Only remove optional files in this checkpoint generation.
            try:
                for optional in [
                    path / "monitor-state.json",
                    *path.glob(".monitor-state.json.*.tmp"),
                ]:
                    optional.unlink(missing_ok=True)
            except OSError:
                pass

    def _checkpoint_state(self, path):
        pending, phases = self.pending, self.phases
        if pending is None:
            rows = self.runtime_history.snapshot(self.owner.applied_version)["updates"]
            if rows:
                pending = rows[-1]
                phases = [
                    {
                        **row["config"],
                        "phase": row["phase"],
                        "seconds": row["seconds"],
                        "profile_id": row["config"].get("profile_id", ""),
                    }
                    for row in pending["measurements"]
                ]
        atomic_json(
            path / "monitor-state.json",
            {
                "format_version": 2,
                "scope": SCOPE,
                "recovery_context": self._recovery_context,
                "measurement_id": self.measurement_id,
                "series_id": self._series_id,
                "data_cursor": self.owner.consumer.cursor,
                "weights_version": self.owner.applied_version,
                "context": self.runtime_history.context,
                "pending": pending,
                "completed": self.pending is None,
                "measurements": phases,
            },
        )

    def complete_update(self):
        if self.owner.active is not None:
            raise RuntimeError(
                "Monitor completion requires the end of the native update"
            )
        if self.owner.published_version != self.owner.applied_version:
            raise RuntimeError("Monitor completion requires acknowledged publication")
        self.completed_tokens = self.owner.data_progress["effective_tokens"]
        try:
            self._complete_update()
        finally:
            self.pending = self._decision = self._workload_cache = None
            self.phases = []

    @_optional_metrics
    def _complete_update(self):
        if self.runtime_history is None:
            return
        if self.owner.applied_version == 0:
            self._finish_recovery()
            return
        checkpoint = self.owner.latest_path
        checkpointed = (
            self.owner.latest_manifest is not None
            and self.owner.latest_manifest["global_step"] == self.owner.applied_version
        )
        prior = self.runtime_history.snapshot(self.owner.applied_version)["updates"]
        if not any(
            row["weights_version"] == self.owner.applied_version for row in prior
        ):
            if self.pending is None:
                # A new measurement series starts at the restored version; it
                # cannot fabricate a pre-change decision or a timing label.
                self._finish_recovery()
                return
            state = {
                "context": self.runtime_history.context,
                "pending": self.pending,
                "measurements": self.phases,
            }
            pending = state["pending"]
            if (
                pending is None
                or pending["target_version"] != self.owner.applied_version
            ):
                raise ValueError(
                    "Recovered monitor checkpoint has no matching update identity"
                )
            measurements = [
                {
                    "phase": row["phase"],
                    "config": {
                        "world_size": row["world_size"],
                        "profile_id": row["profile_id"],
                    },
                    "seconds": row["seconds"],
                    "workload": pending["workload"],
                }
                for row in state["measurements"]
            ]
            record = {
                **pending,
                "observation_sequence": self._next_sequence(),
                "weights_version": self.owner.applied_version,
                "actor_generation": self.owner.generation,
                "update_id": self.owner.consumer.cursor["chain_sha256"],
                "checkpoint_id": checkpoint.name if checkpointed else None,
                "checkpointed": checkpointed,
                "measurements": measurements,
                "timing_scope": SCOPE,
                "missing_phases": [
                    phase
                    for phase in ("train", "checkpoint", "publish")
                    if (phase != "checkpoint" or checkpointed)
                    and phase not in {row["phase"] for row in measurements}
                ],
                "data_cursor": self.owner.consumer.cursor,
            }
            self.runtime_history.append(record)
            self.history.add_workload(BatchWorkload(**record["workload"]))
            for row in measurements:
                self.history.add_measurement(
                    Measurement(
                        row["phase"],
                        row["config"]["world_size"],
                        row["seconds"],
                        row["workload"]["sequence_tokens"],
                        row["config"]["profile_id"],
                    )
                )
        self._finish_recovery()

    def _checkpoint_record(self):
        path = (
            None
            if self.owner.latest_path is None
            else self.owner.latest_path / "monitor-state.json"
        )
        if path is None or not path.exists():
            return None
        state = json.loads(path.read_text())
        if not isinstance(state, dict):
            raise ValueError("Invalid Monitor checkpoint shape")
        if (
            state.get("format_version") != 2
            or state.get("scope") != SCOPE
            or state.get("recovery_context") != self._recovery_context
            or state.get("measurement_id") != self.measurement_id
        ):
            return None  # Different measurements do not invalidate model recovery.
        if (
            state.get("weights_version") != self.owner.applied_version
            or state.get("data_cursor") != self.owner.consumer.cursor
        ):
            raise ValueError("Monitor checkpoint disagrees with recovered progress")
        series = state["series_id"]
        if series != "initial" and (
            not isinstance(series, str)
            or len(series) != 64
            or any(c not in "0123456789abcdef" for c in series)
        ):
            raise ValueError("Invalid measurement series identity")
        return state

    def _pending_record(self, state):
        if self.pending_path.exists() and not state["completed"]:
            pending = json.loads(self.pending_path.read_text())
            if not isinstance(pending, dict):
                raise ValueError("Invalid pending Monitor shape")
            if pending.get("weights_version") == self.owner.applied_version:
                if any(
                    pending.get(key) != state.get(key)
                    for key in (
                        "format_version",
                        "recovery_context",
                        "measurement_id",
                        "series_id",
                        "data_cursor",
                        "context",
                        "pending",
                    )
                ):
                    raise ValueError(
                        "Pending monitor state differs from the checkpoint"
                    )
                return pending
        return state

    def _hydrate_history(self):
        self.history = HistoryWindow()
        window = self.runtime_history.snapshot(self.owner.applied_version)
        records = sorted(
            [*window["updates"], *window["boundaries"]],
            key=lambda row: row["observation_sequence"],
        )
        self._history_sequence = max(
            (row["observation_sequence"] for row in records), default=0
        )
        for record in records:
            if "update_id" in record:
                self.history.add_workload(BatchWorkload(**record["workload"]))
            for row in record["measurements"]:
                workload = row["workload"]
                self.history.add_measurement(
                    Measurement(
                        row["phase"],
                        row["config"]["world_size"],
                        row["seconds"],
                        0 if workload is None else workload["sequence_tokens"],
                        row["config"].get("profile_id", ""),
                    )
                )

    def recover(self):
        """Restore logical progress first; measurements are independently optional."""
        self.completed_tokens = self.owner.data_progress["effective_tokens"]
        self.metrics_available = True
        self.last_error = None
        self.pending = self._decision = self._workload_cache = None
        self.phases = []
        self._recovery_phases = []
        self._recovering = True
        self._recover_metrics()

    @_optional_metrics
    def _recover_metrics(self):
        state = self._checkpoint_record()
        self._series_id = (
            state["series_id"]
            if state is not None
            else (
                "initial"
                if self.owner.latest_path is None
                else json_digest(
                    {
                        "checkpoint": self.owner.latest_path.name,
                        "measurement_id": self.measurement_id,
                        "recovery_context": self._recovery_context,
                    }
                )
            )
        )
        self.runtime_history = RuntimeHistory(
            self.owner.root
            / "monitor"
            / self.measurement_id
            / self._series_id
            / "runtime-history.json",
            context={
                "scope": SCOPE,
                "measurement_id": self.measurement_id,
                "series_id": self._series_id,
                "queue_id": self.owner.inbox.queue.queue_id,
                "checkpoint_root": str(self.owner.store.root),
                "compatibility": self.owner.compatibility,
            },
        )
        self.runtime_history.rollback(self.owner.applied_version)
        self.runtime_history.validate_version(self.owner.applied_version)
        if state is not None:
            if state["context"] != self.runtime_history.context:
                raise ValueError("Monitor checkpoint belongs to another series")
            state = self._pending_record(state)
            self.pending, self.phases = state["pending"], state["measurements"]
        self._hydrate_history()

    @_optional_metrics
    def released(self, seconds, gpu_seconds):
        if self.runtime_history is None:
            return
        if (
            self.owner.published_version != self.owner.applied_version
            or self.owner.active is not None
            and (
                self.owner.latest_manifest is None
                or self.owner.latest_manifest["global_step"]
                != self.owner.applied_version
            )
        ):
            return
        rows = self.runtime_history.snapshot(self.owner.applied_version)["updates"]
        workload = next(
            (
                row["workload"]
                for row in reversed(rows)
                if row["weights_version"] == self.owner.applied_version
            ),
            None,
        )
        if self.pending is not None:
            workload = self.pending["workload"]
        if workload is None and self.owner.latest_path is not None:
            state = self._checkpoint_record()
            pending = None if state is None else state["pending"]
            workload = None if pending is None else pending["workload"]
        self.runtime_history.append_boundary(
            {
                "observation_sequence": self._next_sequence(),
                "weights_version": self.owner.applied_version,
                "checkpoint_id": None
                if self.owner.latest_path is None
                else self.owner.latest_path.name,
                "actor_generation": self.owner.generation,
                "actor_gpu_seconds": gpu_seconds,
                "measurements": [
                    {
                        "phase": "release",
                        "seconds": seconds,
                        "config": {"world_size": self.actor_world_size},
                        "workload": workload,
                    }
                ],
            }
        )
        self.history.add_measurement(
            Measurement(
                "release",
                self.actor_world_size,
                seconds,
                0 if workload is None else workload["sequence_tokens"],
            )
        )
