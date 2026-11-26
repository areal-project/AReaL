# SPDX-License-Identifier: Apache-2.0

"""Run Optimizer revision on frozen CPU replay windows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from areal.autopilot.revision.agent import CommandRevisionAgent, FixedProposals
from areal.autopilot.revision.contracts import Proposal, digest
from areal.autopilot.revision.loop import RevisionLoop
from areal.elastic.reconfigurator.evaluator import (
    Catalog,
    Cutover,
    Observation,
    ReplayEvaluator,
    code_revision,
    load_profiles,
)
from areal.elastic.reconfigurator.evaluator.workload import (
    Backlog,
    EpisodeView,
    ExecutionConfig,
    HistoricalReplay,
    HistoryReplayConfig,
    InflightCall,
    OnlineObservation,
    ServingSLO,
    Turn,
    WorkloadPattern,
)
from areal.elastic.reconfigurator.optimizer import (
    DEFAULT_EVALUATION_BUDGET,
    Layout,
    Optimizer,
    RoleLayout,
    SearchSpace,
)
from areal.elastic.reconfigurator.trigger import WorkloadChange

from .optimizer_reward import WindowResult, optimizer_reward


@dataclass(frozen=True)
class ReplayWindow:
    name: str
    observation: Observation
    workload: WorkloadChange


class OptimizerReplay:
    def __init__(
        self,
        evaluator: ReplayEvaluator,
        grid: SearchSpace,
        windows: tuple[ReplayWindow, ...],
        *,
        search_limit: int,
        context_sha256: str,
    ):
        if not windows or len({window.name for window in windows}) != len(windows):
            raise ValueError("Supply nonempty replay windows with unique names")
        if type(search_limit) is not int or search_limit < 1:
            raise ValueError("search_limit must be a positive integer")
        self.evaluator, self.grid, self.windows = evaluator, grid, windows
        self.context_sha256 = context_sha256
        self.search_limit = search_limit
        self.settings_schema = {
            "max_evaluations": {"minimum": 1, "maximum": search_limit},
        }

    def normalize(self, settings: dict) -> dict:
        if settings.keys() - self.settings_schema.keys():
            raise ValueError("Proposal contains unsupported search settings")
        budget = settings.get("max_evaluations", self.search_limit)
        if type(budget) is not int or not 1 <= budget <= self.search_limit:
            raise ValueError("max_evaluations exceeds the fixed search allowance")
        return {"max_evaluations": budget}

    def evaluate(self, settings: dict) -> dict:
        optimizer = Optimizer.from_grid(
            self.grid, self.evaluator, **self.normalize(settings)
        )
        rows = []
        for window in self.windows:
            observation = window.observation
            decision = optimizer.decide(
                observation, observation.current, confirmed=window.workload
            )
            chosen = next(
                (p for p in decision.predictions if p.layout == decision.target), None
            )
            if chosen is None:
                raise ValueError(f"{window.name}: no scored layout ({decision.reason})")
            row = WindowResult.from_prediction(
                window.name, optimizer.target, chosen, len(decision.predictions)
            )
            rows.append(
                {
                    "metrics": asdict(row),
                    "layout": {
                        side: asdict(getattr(decision.target, side))
                        | {"options": dict(getattr(decision.target, side).options)}
                        for side in ("actor", "rollout")
                    },
                    "reason": decision.reason,
                }
            )
        return {"windows": rows}

    @staticmethod
    def reward(candidate: dict, incumbent: dict):
        return optimizer_reward(
            tuple(WindowResult(**row["metrics"]) for row in candidate["windows"]),
            tuple(WindowResult(**row["metrics"]) for row in incumbent["windows"]),
        )


def _layout(config: ExecutionConfig) -> Layout:
    return Layout(
        RoleLayout(config.actor_backend, config.actor_gpus),
        RoleLayout(config.rollout_backend, config.rollout_gpus),
    )


def _window(item: dict, current: Layout) -> ReplayWindow:
    raw = item["pattern"]
    templates = tuple(
        EpisodeView(
            **{
                **episode,
                "completed_turns": tuple(
                    Turn(**turn) for turn in episode["completed_turns"]
                ),
            }
        )
        for episode in raw["templates"]
    )
    period = raw["period_seconds"]
    arrivals = tuple(tuple(arrival) for arrival in raw["arrivals"])
    if (
        not isfinite(period)
        or period <= 0
        or any(not 0 <= offset < period for offset, _ in arrivals)
    ):
        raise ValueError("Workload arrivals must fall within a finite positive period")
    pattern = WorkloadPattern(period, arrivals, templates)
    view = OnlineObservation(episodes=templates, **item["view"])
    backlog = None
    if item.get("backlog") is not None:
        pending = dict(item["backlog"])
        pending["shapes"] = {
            name: tuple(Turn(**turn) for turn in turns)
            for name, turns in pending["shapes"].items()
        }
        for key in ("queued", "tools"):
            pending[key] = tuple(tuple(row) for row in pending.get(key, ()))
        pending["running"] = tuple(
            InflightCall(**row) for row in pending.get("running", ())
        )
        pending["ready"] = tuple(pending.get("ready", ()))
        if pending.get("actor_event") is not None:
            pending["actor_event"] = tuple(pending["actor_event"])
        backlog = Backlog(**pending)
    observation = Observation(view, current, backlog, item.get("search_ready_at"))
    change = WorkloadChange(("frozen_window",), len(arrivals) / period, {}, pattern)
    return ReplayWindow(item["name"], observation, change)


def load_replay(path: Path) -> OptimizerReplay:
    """Load native model profiles and explicit directed cutover costs."""
    spec = json.loads(path.read_text())
    catalog = Catalog(path.parent / spec["catalog"])
    models = load_profiles(
        **spec["profiles"],
        bandwidth_GBps=spec["bandwidth_GBps"],
        catalog=catalog,
    )
    execution = ExecutionConfig(**spec["execution"])
    current = _layout(execution)
    cutovers = {}
    for row in spec["cutovers"]:
        target = _layout(ExecutionConfig(**row["target"]))
        costs = Cutover(seconds=row["seconds"], gpu_seconds=row["gpu_seconds"])
        if (
            target == current
            or target in cutovers
            or any(not isfinite(value) or value < 0 for value in asdict(costs).values())
        ):
            raise ValueError(
                "Expected unique noncurrent layouts with finite cutover costs"
            )
        cutovers[target] = costs
    evaluator = ReplayEvaluator(
        models,
        HistoricalReplay(HistoryReplayConfig(spec["forecast_seconds"])),
        execution,
        ServingSLO(**spec["slo"]),
        batch_size=spec["batch_size"],
        concurrency=spec["concurrency"],
        staleness=spec["staleness"],
        cutovers={(current, target): costs for target, costs in cutovers.items()},
    )
    grid = (
        SearchSpace(spec["gpu_capacity"], evaluator.batch_size, **spec["grid"])
        if "grid" in spec
        else SearchSpace.for_pool(spec["gpu_capacity"], evaluator.batch_size)
    )
    context = digest(
        {
            "suite": spec,
            "reference_code": code_revision(),
            "revision_code": [
                file.read_text()
                for file in (
                    Path(__file__),
                    Path(__file__).with_name("optimizer_reward.py"),
                )
            ],
            "profile_ids": models.source_ids,
        }
    )
    return OptimizerReplay(
        evaluator,
        grid,
        tuple(_window(window, current) for window in spec["windows"]),
        search_limit=spec.get("search_limit", DEFAULT_EVALUATION_BUDGET),
        context_sha256=context,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--rounds", type=int, help="Total proposal budget for this output directory"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--proposals", type=Path, help="JSON array of fixed proposals")
    source.add_argument(
        "--agent-command", nargs=argparse.REMAINDER, help="Agent argv, passed last"
    )
    args = parser.parse_args(argv)
    if args.proposals is not None:
        proposals = tuple(
            Proposal(**item) for item in json.loads(args.proposals.read_text())
        )
        agent = FixedProposals(proposals)
        rounds = len(proposals) if args.rounds is None else args.rounds
        if rounds > len(proposals):
            parser.error("--rounds exceeds the number of fixed proposals")
    else:
        if args.rounds is None:
            parser.error("--agent-command requires --rounds")
        agent = CommandRevisionAgent(tuple(args.agent_command))
        rounds = args.rounds
    replay = load_replay(args.suite)
    loop = RevisionLoop(agent, replay, args.output, reference={})
    state = loop.run(max_rounds=rounds)
    print(
        json.dumps(
            {
                "output": str(loop.store.path),
                "attempted": len(state["history"]),
                "accepted_round": state["accepted_round"],
                "active_settings": loop.store.incumbent(state)["settings"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
