# SPDX-License-Identifier: Apache-2.0

"Native trajectory metadata, training phases and Online observations."

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import replace

from areal.elastic.contracts import (
    ActorView,
    EpisodeView,
    OnlineLayout,
    OnlineObservation,
    Turn,
)
from areal.utils import logging
from areal.utils.atomic import atomic_json

logger = logging.getLogger("RLTrainer")


class TrainerMonitor:
    """Observe native exports and published updates without consuming their tensors."""

    def __init__(self, owner):
        self.owner = owner
        self.online_observations = NativeOnlineObservations(owner)
        self.metrics_available = True
        self.history = []
        self._drafts = OrderedDict()
        self._episodes = OrderedDict()
        self._consumed = set()
        self._lock = threading.RLock()
        self.active_batch = ()
        self.phase_seconds = {}
        self.execution_id = None

    def bind(self, execution_id):
        self.execution_id = execution_id

    def mark_unavailable(self, error):
        self.metrics_available = False
        logger.warning("Native observations unavailable: %s", error)

    def record_trajectory(self, trajectory, metadata):
        with self._lock:
            self._drafts[id(trajectory)] = dict(metadata)
            while len(self._drafts) > 2048:
                self._drafts.popitem(last=False)

    def begin_update(self, batch):
        with self._lock:
            identities = []
            for trajectory in batch:
                row = self._drafts.pop(id(trajectory), None)
                if row is None:
                    continue
                identity = row["episode_id"]
                self._episodes[identity] = row
                identities.append(identity)
            self.active_batch = tuple(identities)

    def record_phase(self, name, seconds):
        self.phase_seconds[name] = self.phase_seconds.get(name, 0.0) + seconds

    def complete_update(self):
        with self._lock:
            self._consumed.update(self.active_batch)
            while len(self._episodes) > 1024:
                identity, _ = self._episodes.popitem(last=False)
                self._consumed.discard(identity)
            self.history.append(
                {
                    "weights_version": self.owner.applied_version,
                    "episode_ids": self.active_batch,
                    "phase_seconds": dict(self.phase_seconds),
                    "execution_id": self.execution_id,
                }
            )
            self.history = self.history[-256:]
            try:
                atomic_json(self.owner.root / "native-updates.json", self.history)
            except OSError as error:
                self.mark_unavailable(error)
            self.active_batch = ()
            self.phase_seconds = {}

    def snapshot(self, now):
        owner = self.owner
        origin = self.online_observations.origin_wall
        with self._lock:
            rows = list(self._episodes.values())
            rows.sort(key=lambda r: r["episode_id"] not in self._consumed)
            episodes = tuple(
                EpisodeView(
                    r["episode_id"],
                    "native-trajectory-aggregate",
                    max(0.0, r["available_wall_time"] - origin),
                    True,
                    max(0.0, r["completed_wall_time"] - origin),
                    (Turn(r["input_tokens"], r["output_tokens"]),),
                    None,
                    None,
                )
                for r in rows
            )
            consumed = sum(r["episode_id"] in self._consumed for r in rows)
            active = self.active_batch
        positions = {
            device: index for index, device in enumerate(owner.resource_reservation)
        }
        leases = [
            (1, "rollout", tuple((0, positions[d]) for d in owner.rollout_devices))
        ]
        if owner.actor is not None:
            leases.insert(
                0,
                (
                    0,
                    "actor",
                    tuple(
                        (0, positions[d])
                        for d in owner.actor_devices[: owner.actor_world_size]
                    ),
                ),
            )
        return OnlineObservation(
            now=now,
            layout=self.online_observations.layout(),
            episodes=episodes,
            requests=(),
            services=(),
            actor=ActorView(active, 0, now, None, owner.actor is not None),
            consumed_episodes=consumed,
            training_version=owner.applied_version,
            provider_version=owner.published_version,
            busy=not owner.control_snapshot().boundary_ready,
            input_closed=False,
            leases=tuple(leases),
        )


class NativeOnlineObservations:
    def __init__(self, owner):
        self.owner = owner
        path = owner.root / "online-observation-origin.json"
        if path.exists():
            self.origin_wall = json.loads(path.read_text())["wall_time"]
        else:
            self.origin_wall = time.time()
            atomic_json(path, {"wall_time": self.origin_wall})
        self.origin_monotonic = time.monotonic() - (time.time() - self.origin_wall)

    def now(self, monotonic=None):
        return max(
            0.0,
            (time.monotonic() if monotonic is None else monotonic)
            - self.origin_monotonic,
        )

    def layout(self):
        from areal.api.alloc_mode import ModelAllocation

        actor = self.owner.allocation.parallel
        rollout = ModelAllocation.from_str(self.owner.config.rollout.backend).parallel
        return OnlineLayout(
            actor.dp_size,
            actor.tp_size,
            len(self.owner.rollout_devices) // rollout.tp_size,
            rollout.tp_size,
        )

    def observe(self, *, now=None):
        return self.owner.monitor.snapshot(self.now(now))

    def validate(self, view):
        if not isinstance(view, OnlineObservation):
            raise TypeError("Online observation provider must return OnlineObservation")
        owner = self.owner
        if (
            view.layout != self.layout()
            or view.training_version != owner.applied_version
            or view.provider_version != owner.published_version
        ):
            raise ValueError("Online observation does not match the native boundary")
        return replace(
            view,
            busy=view.busy or not owner.control_snapshot().boundary_ready,
        )
