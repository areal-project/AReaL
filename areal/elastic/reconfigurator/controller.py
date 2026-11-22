# SPDX-License-Identifier: Apache-2.0

"""Run the Reference policy beside online PPO and submit at publication boundaries."""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock, Thread
from uuid import uuid4

from areal.utils import logging
from areal.v2.reconfiguration import Device, ExecutionLayout, RayDevicePool

from .evaluator import Observation, ReplayEvaluator
from .evaluator.cost_model.contracts import ActorConfig
from .evaluator.cost_model.loader import Catalog
from .evaluator.cost_model.models import AnalyticalCostModels
from .evaluator.cost_model.reconfiguration.model import (
    ActorStages,
    IOBandwidth,
    ReconfigurationModel,
    RolloutStages,
)
from .evaluator.workload import (
    ExecutionConfig,
    HistoricalReplay,
    HistoryReplayConfig,
    ServingSLO,
)
from .observation import LiveObservation
from .optimizer import Layout, Optimizer, RoleLayout
from .trigger import WorkloadTrigger

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


def _policy(trainer, binding, warm_actor, warm_rollout, parameters, models):
    cfg = trainer.config
    settings = cfg.elastic
    stages = parameters["cutover_parameters"]
    if any(stages[k] != parameters["profiles"][k] for k in ("hardware", "model")):
        raise ValueError(
            "Cutover parameters must match the hardware and model profiles"
        )
    shape = models.actor_model.model
    weight = models.weight_bytes / 1e9
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
    active = set(actor_layout.devices + rollout_layout.devices)
    ordered = (
        rollout_layout.devices
        + tuple(device for device in pool if device not in active)
        + actor_layout.devices
    )

    def place(layout):
        actors, rollouts = ordered[-layout.actor.gpus :], ordered[: layout.rollout.gpus]
        replica_size = (
            len(rollouts)
            // ExecutionConfig(
                template.actor_backend, layout.rollout.backend
            ).rollout_replicas
        )
        for start in range(0, len(rollouts), replica_size):
            if len({d.node for d in rollouts[start : start + replica_size]}) != 1:
                raise ValueError("An SGLang instance must fit on one node")
        return actors, rollouts

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
        self.models = AnalyticalCostModels.from_profiles(
            **self.parameters["profiles"],
            catalog=Catalog(Path(self.settings.catalog)),
            bandwidth_GBps=self.parameters["publication_GBps"],
            actor_runtime=ActorConfig.from_areal(asdict(trainer.config.actor)),
        )
        self.trigger = WorkloadTrigger()
        self.lock = Lock()
        self.stop = Event()
        self.warm_actor = set()
        self.warm_rollout = set()
        self.binding = self._binding()
        self.policy = self._bind(self.binding)
        self.version = trainer.actor.get_version()
        self.actor_events = []
        self.generation = 0
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
        running = queued = None
        while not self.stop.is_set():
            with self.lock:
                version, policy = self.version, self.policy
            view = self.reader.snapshot(version)
            change = self.trigger.observe(view) if view is not None else None
            if change is not None and (
                change.pattern.templates or not change.pattern.arrivals
            ):
                with self.lock:
                    if policy is not self.policy:
                        continue
                    events = tuple(self.actor_events)
                    self.generation += 1
                    self.ready = None
                    generation = self.generation
                self.trigger.searched(change)
                backlog = self.reader.backlog(view, change.pattern, events)
                queued = (generation, policy, view, change, backlog)
            if running is not None and running[0].done():
                future, epoch, source, started = running
                decision = future.result()
                logger.info(
                    "Elastic decision %s: %s -> %s (%s, %d candidates)",
                    epoch,
                    decision.source,
                    decision.target,
                    decision.reason,
                    len(decision.predictions),
                )
                with self.lock:
                    if epoch == self.generation and source is self.policy:
                        self.ready = (source, decision, started)
                running = None
            if running is None and queued is not None:
                epoch, source, snapshot, change, backlog = queued
                queued = None
                with self.lock:
                    if source is not self.policy:
                        continue
                optimizer, layout, _ = source
                started = time.monotonic()
                ready_at = (
                    time.time()
                    - self.reader.origin
                    + self.settings.estimated_search_seconds
                )
                observation = Observation(snapshot, layout, backlog, ready_at)
                future = worker.submit(
                    optimizer.decide, observation, layout, confirmed=change
                )
                running = (future, epoch, source, started)
            self.stop.wait(self.settings.poll_seconds)

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
            if self.ready is None:
                return
            source, decision, started = self.ready
            self.ready = None
            if (
                source is not self.policy
                or time.monotonic() - started >= self.settings.forecast_seconds
            ):
                return
            _, layout, place = source
            if decision.source != layout or decision.target == layout:
                return

        target = decision.target
        actors, rollouts = place(target)
        actor_layout = None
        rollout_layout = None
        state_path = None
        if target.actor != layout.actor:
            actor_layout = ExecutionLayout(target.actor.backend, actors)
            state_path = str(
                self.directory / f"actor-state-{self.version}-{uuid4().hex}"
            )
        if target.rollout != layout.rollout:
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
        logger.info(
            "Automatic elastic published layout at version %s: %s",
            self.version,
            binding,
        )

    def close(self):
        self.stop.set()
        self.thread.join()
