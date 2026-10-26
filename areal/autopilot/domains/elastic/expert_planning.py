# SPDX-License-Identifier: Apache-2.0

"""Compose Basic joint search with the independently frozen Expert simulator."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType

from areal.elastic.contracts import BatchWorkload, WorkloadStats
from areal.elastic.policy.basic.joint import JointConfig, JointPrediction
from areal.elastic.policy.reference.catalog import numerical_environment
from areal.elastic.policy.reference.core import Estimate, fingerprint
from areal.elastic.policy.reference.scenario import (
    ExecutionConfig,
    ExpertResult,
    ExternalCall,
    Group,
    PlanStep,
    Request,
    Scenario,
)
from areal.elastic.policy.reference.simulator import ExpertSimulator


def expert_groups_from_workload(
    episodes: tuple[dict, ...], *, tool_seconds_per_unit: dict[str, float]
) -> tuple[Group, ...]:
    """Experiment adapter: individual trajectories, one sample/group, all accepted.

    Tool service rates are explicit experiment inputs. Pool capacities, hardware,
    model and admission limits belong to Scenario, never the neutral trace.
    """
    groups = []
    for episode in episodes:
        nodes, lengths, loss = [], [], 0
        for node in episode["nodes"]:
            predecessors = tuple(node["predecessors"])
            if node["kind"] == "inference":
                nodes.append(
                    Request(
                        node["id"],
                        node["input_tokens"],
                        node["output_tokens"],
                        predecessors,
                    )
                )
                lengths.append(node["input_tokens"] + node["output_tokens"])
                loss += node["output_tokens"]
            elif node["kind"] == "tool":
                reference = node["work_reference"]
                rate = tool_seconds_per_unit[reference]
                if rate <= 0:
                    raise ValueError("Tool seconds/unit must be positive")
                nodes.append(
                    ExternalCall(
                        node["id"],
                        node["tool_type"],
                        Estimate(
                            node["work_units"] * rate,
                            "assumed",
                            source_ids=(reference,),
                        ),
                        predecessors,
                    )
                )
            else:
                raise ValueError("Unknown workload node kind")
        groups.append(
            Group(
                episode["episode_id"],
                tuple(nodes),
                WorkloadStats(
                    len(lengths),
                    sum(lengths),
                    loss_tokens=loss,
                    sequence_lengths=tuple(lengths),
                ),
                1,
                available_at=episode["available_at"],
                task_id=episode["task_id"],
            )
        )
    return tuple(groups)


@dataclass(frozen=True)
class ExpertSearchBatch:
    """Work budget only; the original Scenario determines actual batch selection."""

    actor: BatchWorkload


@dataclass(frozen=True)
class ExpertJointWindow:
    scenario: Scenario
    current: JointConfig
    steps: tuple[PlanStep, ...]
    reservation: tuple[str, ...]
    source_id: str
    batches: tuple[ExpertSearchBatch, ...] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "reservation", tuple(self.reservation))
        scenario = self.scenario
        if (
            not self.source_id
            or len(set(self.reservation)) != len(self.reservation)
            or not scenario.initial.complete
            or scenario.dynamic_batch
            or not self.steps
            or len(scenario.groups) != len(self.steps) * scenario.groups_per_batch
            or any(not g.accepted for g in scenario.groups)
            or any(not s.update_succeeds or not s.publish_succeeds for s in self.steps)
        ):
            raise ValueError(
                "Expert search requires a complete, fixed successful workload window"
            )
        batches = []
        for offset in range(0, len(scenario.groups), scenario.groups_per_batch):
            groups = scenario.groups[offset : offset + scenario.groups_per_batch]
            if any(g.workload.loss_tokens is None for g in groups):
                raise ValueError(
                    "Explicit loss-token counts are required for search accounting"
                )
            if any(
                int(value) != value
                for g in groups
                for value in (g.workload.loss_tokens, g.workload.sequence_tokens)
            ):
                raise ValueError("Each group requires exact integer token counts")
            loss = sum(g.workload.loss_tokens for g in groups)
            tokens = sum(g.workload.sequence_tokens for g in groups)
            if int(loss) != loss or int(tokens) != tokens:
                raise ValueError("Search token counts must be exact integers")
            batches.append(
                ExpertSearchBatch(
                    BatchWorkload(
                        sum(g.samples for g in groups),
                        int(loss),
                        int(tokens),
                    )
                )
            )
        object.__setattr__(self, "batches", tuple(batches))


class ExpertJointModel:
    """JointOptimizer's model port; no fitting, search or runtime execution inside Expert.

    Every query keeps the full request DAG and plan options, replacing only the
    explicitly mapped execution config. Request-level results remain available
    in results; JointPrediction is only the search objective projection.
    """

    def __init__(
        self,
        simulator: ExpertSimulator,
        bindings: tuple[tuple[JointConfig, ExecutionConfig], ...],
    ):
        self.simulator = simulator
        self.bindings = MappingProxyType(dict(bindings))
        if not self.bindings or len(self.bindings) != len(bindings):
            raise ValueError("Explicit unique joint-to-Expert bindings are required")
        for config, execution in bindings:
            if (
                config.actor.profile_id
                != simulator.models.actor.profile_key(execution.actor)
                or config.rollout.model_id
                != fingerprint(simulator.models.rollout.model)
                or config.rollout.hardware_id
                != fingerprint(simulator.models.rollout.hardware)
                or config.rollout.runtime_id
                != simulator.models.rollout.runtime_revision
            ):
                raise ValueError(
                    "Joint alias must identify its frozen Expert model/profile"
                )
            if config.actor.world_size != execution.actor.parallel.world_size or any(
                getattr(config.rollout, key) != getattr(execution.rollout, key)
                for key in (
                    "replicas",
                    "tp",
                    "backend",
                    "max_batch_sequences",
                    "max_batch_tokens",
                    "kv_capacity_tokens",
                )
            ):
                raise ValueError(
                    "Joint alias disagrees with the complete Expert configuration"
                )
        self.model_snapshot = simulator.models.snapshot_id
        self.numerical_environment = numerical_environment()
        self.snapshot_id = fingerprint(
            self.model_snapshot, bindings, self.numerical_environment
        )
        self.results: dict[str, ExpertResult] = {}

    @staticmethod
    def query_id(window: ExpertJointWindow, config: JointConfig) -> str:
        return fingerprint(window, config)

    def evaluate(
        self, window: ExpertJointWindow, config: JointConfig
    ) -> JointPrediction:
        if (
            self.simulator.models.snapshot_id != self.model_snapshot
            or numerical_environment() != self.numerical_environment
        ):
            raise ValueError("Expert model changed during a frozen search")
        execution = self.bindings.get(config)
        source = self.bindings.get(window.current)
        if source != window.scenario.initial.config:
            raise ValueError(
                "Search current configuration differs from Expert initial state"
            )
        if execution is None or not set(
            execution.actor_devices
            + execution.rollout_devices
            + source.actor_devices
            + source.rollout_devices
        ) <= set(window.reservation):
            return JointPrediction(
                "unsupported", config, reason="No in-reservation Expert binding"
            )
        steps = tuple(replace(step, config=execution) for step in window.steps)
        result = self.simulator.simulate(window.scenario, steps)
        self.results[self.query_id(window, config)] = result
        consumed = [
            gid
            for event in result.events
            if event.kind == "actor_start"
            for gid in dict(event.details)["groups"]
        ]
        groups = {g.group_id: g for g in window.scenario.groups}
        status, reason = result.status, result.reason
        if status == "complete" and (
            result.completed_updates != len(steps)
            or len(consumed) != len(set(consumed))
            or set(consumed) != groups.keys()
            or result.remaining_devices
            or result.running_groups
            or result.queued_groups
            or result.cancelled_groups
        ):
            status, reason = (
                "incomplete",
                "Expert did not consume the identical task window and cleanup",
            )
        if result.queue_wait_seconds is None:
            raise ValueError("Expert did not report queue-wait accounting")
        return JointPrediction(
            status,
            config,
            result.seconds,
            result.actor_gpu_seconds,
            result.rollout_gpu_seconds,
            result.completed_updates,
            sum(groups[gid].samples for gid in consumed),
            sum(int(groups[gid].workload.loss_tokens) for gid in consumed),
            sum(int(groups[gid].workload.sequence_tokens) for gid in consumed),
            result.queue_wait_seconds,
            reason,
            assumptions=result.notes
            + (
                f"Expert evidence: {result.evidence}; full request events retained separately",
                "Scenario concurrency/routing defines admission; Basic max_inflight_batches is not reapplied",
            ),
            source_ids=(window.source_id, self.snapshot_id, result.snapshot_id),
        )
