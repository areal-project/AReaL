# SPDX-License-Identifier: Apache-2.0

"""Run the Reference policy beside online PPO and submit at publication boundaries."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import cache
from itertools import combinations
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

from areal.utils import logging
from areal.v2.reconfiguration import (
    Device,
    ExecutionLayout,
    RayDevicePool,
    sglang_gpus_per_node,
)

from .evaluator import Catalog, Observation, ReplayEvaluator, load_profiles
from .evaluator.actor.cost_model import actor_config
from .evaluator.areal.cost_model import (
    ActorStages,
    IOBandwidth,
    ReconfigurationModel,
    RolloutStages,
)
from .evaluator.contracts import ActorConfig
from .evaluator.workload import (
    ExecutionConfig,
    HistoricalReplay,
    HistoryReplayConfig,
    ServingSLO,
)
from .observation import LiveObservation
from .optimizer import Layout, Optimizer, RoleLayout
from .trigger import StableWorkloadTrigger

logger = logging.getLogger("ElasticController")


def configure_devices(config, supplied):
    """Set up event forwarding before PPOTrainer constructs its scheduler."""
    settings = config.elastic
    devices = []
    for address in settings.devices:
        node, _, index = address.rpartition(":")
        devices.append(Device(node, int(index)))
    if supplied is not None:
        if devices and tuple(devices) != tuple(supplied):
            raise ValueError(
                "Conflicting automatic and explicit reconfiguration devices"
            )
        devices = list(supplied)
    if not devices:
        raise ValueError("Automatic elastic PPO requires an allocated device list")
    if not all(
        math.isfinite(x) and x > 0
        for x in (settings.poll_seconds, settings.forecast_seconds)
    ):
        raise ValueError("Polling and forecasting periods must be positive")
    if (
        not math.isfinite(settings.estimated_search_seconds)
        or settings.estimated_search_seconds < 0
    ):
        raise ValueError("Expected a nonnegative search duration estimate")
    directory = Path(settings.run_dir).resolve()
    config.rollout.scheduling_spec[0].env_vars["AREAL_ELASTIC_EVENT_DIR"] = str(
        directory
    )
    return devices


def _pack_rollout(
    devices: tuple[Device, ...], tp_size: int, pp_size: int, replicas: int
) -> tuple[Device, ...] | None:
    by_node: dict[str, list[Device]] = defaultdict(list)
    for device in devices:
        by_node[device.node].append(device)
    replica_size = tp_size * pp_size
    slots = tuple(
        device
        for node_devices in by_node.values()
        for start in range(0, len(node_devices) - replica_size + 1, replica_size)
        for device in node_devices[start : start + replica_size]
    )
    required = replicas * replica_size
    if len(slots) >= required:
        return slots[:required]

    devices_by_node = tuple(by_node.values())
    capacities = tuple(map(len, devices_by_node))
    widths = []
    for nnodes in range(1, min(len(by_node), replica_size) + 1):
        try:
            local = sglang_gpus_per_node(tp_size, pp_size, nnodes)
        except ValueError:
            continue
        widths.append((nnodes, local))

    @cache
    def pack(available: tuple[int, ...], count: int) -> tuple[Device, ...] | None:
        if not count:
            return ()
        if sum(available) < count * replica_size:
            return None
        examined = set()
        for nnodes, local in widths:
            eligible = [i for i, capacity in enumerate(available) if capacity >= local]
            for nodes in combinations(eligible, nnodes):
                remainder = list(available)
                for node in nodes:
                    remainder[node] -= local
                # Node identity does not affect feasibility for node-local ranks.
                shape = tuple(sorted(remainder))
                if shape in examined:
                    continue
                examined.add(shape)
                placement = pack(tuple(remainder), count - 1)
                if placement is not None:
                    return (
                        tuple(
                            device
                            for node in nodes
                            for device in devices_by_node[node][
                                capacities[node] - available[node] : capacities[node]
                                - remainder[node]
                            ]
                        )
                        + placement
                    )
        return None

    return pack(capacities, replicas)


def _placement(
    pool: tuple[Device, ...],
    current: Layout,
    actor_layout: ExecutionLayout,
    rollout_layout: ExecutionLayout,
) -> Callable[[Layout], tuple[tuple[Device, ...], tuple[Device, ...]]]:
    actor_devices, rollout_devices = actor_layout.devices, rollout_layout.devices
    active = set(actor_devices + rollout_devices)
    ordered = (
        rollout_devices
        + tuple(device for device in pool if device not in active)
        + actor_devices
    )
    active_actor = set(actor_devices)
    without_actor = tuple(device for device in ordered if device not in active_actor)

    def place(layout: Layout) -> tuple[tuple[Device, ...], tuple[Device, ...]]:
        if layout.gpus > len(pool):
            raise ValueError("Layout exceeds the GPU pool")
        keep_actor = layout.actor == current.actor
        if layout.rollout == current.rollout:
            rollouts = rollout_devices
        else:
            replicas, tp_size, pp_size = ExecutionConfig(
                layout.actor.backend, layout.rollout.backend
            ).rollout_parallel
            rollouts = (
                _pack_rollout(without_actor, tp_size, pp_size, replicas)
                if keep_actor
                else None
            )
            if rollouts is None:
                rollouts = _pack_rollout(ordered, tp_size, pp_size, replicas)
            if rollouts is None:
                raise ValueError("The SGLang replicas do not fit in the GPU pool")
        occupied = set(rollouts)
        if keep_actor and active_actor.isdisjoint(occupied):
            return actor_devices, rollouts
        available = tuple(device for device in ordered if device not in occupied)
        return available[-layout.actor.gpus :], rollouts

    return place


def _policy(trainer, binding, warm_actor, warm_rollout, parameters, models):
    cfg = trainer.config
    settings = cfg.elastic
    stages = parameters["cutover_parameters"]
    if any(stages[k] != parameters["profiles"][k] for k in ("hardware", "model")):
        raise ValueError(
            "Cutover parameters must match the hardware and model profiles"
        )
    shape = models.actor_model.model
    weight = shape.total_parameters * 2 / 1e9
    actor = ActorStages(
        weight,
        6 * weight,
        IOBandwidth(**stages["write_GBps"]),
        IOBandwidth(**stages["read_GBps"]),
        warm_nodes=frozenset(warm_actor),
        expert_fraction=shape.layers
        * shape.expert_parameters_per_layer
        / shape.total_parameters
        if shape.mlp_type == "moe"
        else 0.0,
        **stages["actor"],
    )
    rollout = RolloutStages(warm_nodes=frozenset(warm_rollout), **stages["rollout"])
    actor_layout, rollout_layout, dp = binding
    template = ExecutionConfig(
        actor_layout.backend,
        rollout_layout.backend,
        actor_n_mbs=cfg.actor.mb_spec.n_mbs,
        actor_max_tokens_per_mb=cfg.actor.mb_spec.max_tokens_per_mb,
        actor_pad_to_maximum=getattr(cfg.actor, "pad_to_maximum", False),
        sglang_dp_size=dp,
        sglang_enable_dp_attention=cfg.sglang.enable_dp_attention,
        sglang_max_running_requests=cfg.sglang.max_running_requests,
        sglang_mem_fraction_static=cfg.sglang.mem_fraction_static,
        sglang_max_prefill_tokens=cfg.sglang.max_prefill_tokens,
        rollout_max_concurrent_rollouts=cfg.rollout.max_concurrent_rollouts,
    )
    current = Layout(
        RoleLayout(template.actor_backend, template.actor_gpus),
        RoleLayout(template.rollout_backend, template.rollout_gpus),
    )
    pool = tuple(trainer._reconfiguration_devices)
    place = _placement(pool, current, actor_layout, rollout_layout)

    cut = ReconfigurationModel(
        actor.predict,
        rollout.predict,
        place,
        stages["pause_disconnect"],
        stages["reconnect_resume"],
    )
    window = parameters["window"]
    slo = ServingSLO(
        window["ttft_seconds"], window["tpot_seconds"], window["pressure_fraction"]
    )
    evaluator = ReplayEvaluator(
        models,
        HistoricalReplay(HistoryReplayConfig(settings.forecast_seconds)),
        template,
        slo,
        batch_size=cfg.train_dataset.batch_size,
        concurrency=cfg.rollout.max_concurrent_rollouts,
        staleness=cfg.rollout.max_head_offpolicyness,
        reconfiguration=cut,
    )
    search = Optimizer(
        evaluator=evaluator,
        gpu_capacity=len(pool),
        minimum_request_fraction=slo.minimum_request_fraction,
    )
    return search, current, place


class ElasticController:
    def __init__(self, trainer):
        self.trainer = trainer
        self.settings = trainer.config.elastic
        self.directory = Path(self.settings.run_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        # Event streams belong to a single training invocation. Producers may
        # start first, so reserve ownership separately from creating the directory.
        (self.directory / "trainer-run").mkdir()
        self.reader = LiveObservation(self.directory)
        self.parameters = json.loads(Path(self.settings.settings).read_text())
        self.models = load_profiles(
            **self.parameters["profiles"],
            catalog=Catalog(Path(self.settings.catalog)),
            bandwidth_GBps=self.parameters["publication_GBps"],
            actor_runtime=ActorConfig.from_areal(asdict(trainer.config.actor)),
        )
        self.trigger = StableWorkloadTrigger()
        self.lock = Lock()
        self.stop = Event()
        self.wake = Event()
        self.warm_actor = set()
        self.warm_rollout = set()
        self.binding = self._binding()
        self.policy = self._bind(self.binding)
        self.version = trainer.actor.get_version()
        self.actor_events = []
        self.latest_batch: tuple[str, ...] = ()
        self.switching = False
        self.consumed_sessions = set()
        self.generation = 0
        self.running = None
        self.ready = None
        self.error = None
        self.thread = Thread(target=self._run, name="elastic-observe", daemon=True)
        self.thread.start()

    def _binding(self):
        pool = RayDevicePool.for_scheduler(self.trainer.scheduler)

        def read(role):
            return ExecutionLayout.from_gpu_slots(
                role.config.backend, role.config.gpu_devices, pool
            )

        return (
            read(self.trainer.actor),
            read(self.trainer.rollout),
            self.trainer.config.sglang.dp_size,
        )

    def _bind(self, binding):
        self.warm_actor.update(d.node for d in binding[0].devices)
        self.warm_rollout.update(d.node for d in binding[1].devices)
        return _policy(
            self.trainer,
            binding,
            self.warm_actor,
            self.warm_rollout,
            self.parameters,
            self.models,
        )

    def record(self, kind, count=0):
        with self.lock:
            self.actor_events.append((kind, time.time(), count))

    def on_actor_batch(self, batch):
        at = time.time()
        session_ids = tuple(traj.pop("_elastic_session_id", "") for traj in batch)
        with self.lock:
            self.actor_events.append(("take", at, len(batch)))
            self.latest_batch = session_ids
            self.consumed_sessions.update(sid for sid in session_ids if sid)
        self.wake.set()

    def _run(self):
        try:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="elastic-search"
            ) as worker:
                self._observe(worker)
        except Exception as exc:
            with self.lock:
                self.error = exc
            logger.exception("Automatic elastic search failed")

    def _observe(self, worker):
        next_poll = time.monotonic()
        while not self.stop.is_set():
            self.wake.clear()
            with self.lock:
                self._collect_search()
            if time.monotonic() >= next_poll:
                next_poll = time.monotonic() + self.trigger.window_seconds
                self._poll_trigger(worker)
            self.wake.wait(
                min(self.settings.poll_seconds, max(0, next_poll - time.monotonic()))
            )

    def _poll_trigger(self, worker):
        with self.lock:
            if self.switching or self.running is not None or self.ready is not None:
                return
            at = time.time()
            version, policy = self.version, self.policy
            events = tuple(self.actor_events)
            sessions = self.latest_batch
            consumed = tuple(self.consumed_sessions)
        if not sessions:
            return
        view = self.reader.snapshot(version, at=at)
        if view is None:
            return
        batch = self.reader.episode_ids(sessions)
        with self.lock:
            if self.switching or policy is not self.policy:
                return
            change = self.trigger.observe(view, batch)
        if change is None:
            return

        backlog = self.reader.backlog(
            view,
            change.pattern,
            events,
            consumed=set(self.reader.episode_ids(consumed)),
        )
        optimizer, layout, _ = policy
        if (
            backlog is not None
            and backlog.actor_event is not None
            and backlog.actor_event[0] == "actor"
        ):
            shapes = tuple(ep.completed_turns for ep in change.pattern.templates)
            try:
                execution = actor_config(
                    self.models.actor_template, optimizer.evaluator.template
                )
                duration = self.models.actor_model.estimate_batch(
                    execution, shapes
                ).duration_seconds()
            except ValueError as exc:
                logger.warning("Cannot estimate the active Actor batch: %s", exc)
            else:
                started = next(
                    stamp for kind, stamp, _ in reversed(events) if kind == "take"
                )
                backlog = replace(
                    backlog, actor_event=("actor", max(0.0, duration - (at - started)))
                )

        with self.lock:
            if self.switching or policy is not self.policy:
                return
            ready_at = (
                view.now
                + max(0.0, time.time() - at)
                + self.settings.estimated_search_seconds
            )
            observation = Observation(view, layout, backlog, ready_at)
            future = worker.submit(
                optimizer.decide, observation, layout, confirmed=change
            )
            self.running = (future, self.generation, policy, at, change)
            future.add_done_callback(lambda _: self.wake.set())
        logger.info(
            "Elastic trigger at version %s: %s (%.3f arrivals/s, score=%s)",
            version,
            ", ".join(change.reasons),
            change.rate,
            "initial" if change.score is None else f"{change.score:.1%}",
        )

    def _collect_search(self):
        """Collect under the controller lock, including at the publication boundary."""
        if self.running is None or not self.running[0].done():
            return
        future, epoch, source, at, change = self.running
        self.running = None
        decision = future.result()
        logger.info(
            "Elastic decision %s: %s -> %s (%s, %d candidates)",
            epoch,
            decision.source,
            decision.target,
            decision.reason,
            len(decision.predictions),
        )
        if epoch == self.generation and source is self.policy:
            self.ready = (source, decision, at, change)

    def at_boundary(self):
        """Apply a ready decision synchronously at a PPO publication boundary."""
        binding = self._binding()
        with self.lock:
            self.version = self.trainer.actor.get_version()
            if self.error is not None:
                raise RuntimeError("Automatic elastic search failed") from self.error
            if binding != self.binding:
                self.binding = binding
                self.policy = self._bind(binding)
                self.generation += 1
                self.ready = None
                self.trigger.bind(None)
            self._collect_search()
            if self.ready is None:
                return
            source, decision, started, change = self.ready
            self.ready = None
            if (
                source is not self.policy
                or time.time() - started >= self.settings.forecast_seconds
            ):
                return
            _, layout, place = source
            if decision.source != layout:
                return
            selected = next(
                (p for p in decision.predictions if p.layout == decision.target), None
            )
            if (
                selected is None
                or not selected.supported
                or selected.request_fraction is None
                or selected.gpu_seconds is None
            ):
                return
            if decision.target == layout:
                self.trigger.bind(change)
                return
            self.switching = True

        target = decision.target
        actors, rollouts = place(target)
        actor_layout = None
        rollout_layout = None
        state_path = None
        if target.actor != layout.actor or actors != binding[0].devices:
            actor_layout = ExecutionLayout(target.actor.backend, actors)
            state_path = str(
                self.directory / f"actor-state-{self.version}-{uuid4().hex}"
            )
        if target.rollout != layout.rollout or rollouts != binding[1].devices:
            dp = dict(target.rollout.options).get("sglang_dp_size")
            rollout_layout = ExecutionLayout(
                target.rollout.backend, rollouts, attention_dp_size=dp
            )
        self.trainer._apply_reconfiguration(
            actor_layout=actor_layout,
            rollout_layout=rollout_layout,
            state_path=state_path,
        )
        binding = self._binding()
        with self.lock:
            self.binding = binding
            self.policy = self._bind(binding)
            self.generation += 1
            self.ready = None
            self.trigger.bind(change)
            self.switching = False
        logger.info(
            "Automatic elastic published layout at version %s: %s",
            self.version,
            binding,
        )

    def close(self):
        self.stop.set()
        self.wake.set()
        self.thread.join()
