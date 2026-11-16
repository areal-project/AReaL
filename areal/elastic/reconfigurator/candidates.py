# SPDX-License-Identifier: Apache-2.0

"""Factorized configuration space and v2 probes without component time scoring."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from functools import lru_cache

from areal.elastic.contracts import ExecutionConfig


def distance(left: ExecutionConfig, right: ExecutionConfig) -> int:
    return sum(
        getattr(left, f.name) != getattr(right, f.name) for f in fields(ExecutionConfig)
    )


def grid_roles(grid):
    """Enumerate each side once; never enumerate their Cartesian product."""
    parameters = grid.get("parameters", {})
    common = dict(
        batch_size=grid["batch_size"],
        gpu_budget=grid["gpu_budget"],
        moe=grid.get("moe", False),
    )
    actors = tuple(
        ExecutionConfig.generate(
            **common,
            actor=grid["actor"],
            rollout={},
            parameters={k: v for k, v in parameters.items() if k.startswith("actor_")},
        )
    )
    rollouts = tuple(
        ExecutionConfig.generate(
            **common,
            actor={},
            rollout=grid["rollout"],
            parameters={
                k: v for k, v in parameters.items() if not k.startswith("actor_")
            },
        )
    )
    a = Counter(c.actor_gpus for c in actors)
    r = Counter(c.rollout_gpus for c in rollouts)
    count = sum(
        na * nr
        for ga, na in a.items()
        for gr, nr in r.items()
        if ga + gr <= grid["gpu_budget"]
    )
    return actors, rollouts, count


def restrict_grid(grid: dict, current: ExecutionConfig, roles: dict) -> dict:
    """Freeze unavailable roles exactly as Reference's role-history filter does."""
    result = copy.deepcopy(grid)
    for role, dimensions, values in (
        ("actor", ("dp", "tp", "pp", "cp", "ep", "etp"), current.actor_parallel),
        ("rollout", ("replicas", "tp", "pp"), current.rollout_parallel),
    ):
        if roles[role]:
            continue
        result[role] = {k: [v] for k, v in zip(dimensions, values, strict=True)}
        for key, value in asdict(current).items():
            if key.endswith("_backend"):
                continue
            if key.startswith("actor_") == (role == "actor"):
                result["parameters"][key] = [value]
    return result


def role_witnesses(actors, rollouts, current: ExecutionConfig, budget: int):
    """Let the existing role-history gate see alternatives without a full product."""
    witnesses = [current]
    for options in (
        (actor.with_rollout(current) for actor in actors),
        (current.with_rollout(rollout) for rollout in rollouts),
        (actor.with_rollout(rollout) for actor in actors for rollout in rollouts),
    ):
        for candidate in options:
            if (
                candidate != current
                and candidate.actor_gpus + candidate.rollout_gpus <= budget
            ):
                witnesses.append(candidate)
                break
    return tuple(dict.fromkeys(witnesses))


class HeuristicPreparation:
    """Use observed packing bounds and topology coverage, never timing estimates."""

    def __init__(self):
        self._grids = lru_cache(maxsize=8)(lambda text: grid_roles(json.loads(text)))
        self.costs = None
        self.longest = 0
        self.backlog = False
        self.preparations = 0
        self.validated = 0

    def grid_roles(self, grid):
        return self._grids(json.dumps(grid))

    def prepare(self, evaluator, view, forecasts, actors, rollouts):
        evaluator.prepare_context(view, forecasts)
        self.costs = evaluator.cost_models
        self.longest = max(
            (
                t.input_tokens + t.output_tokens
                for e in evaluator._known_batch
                for t in e
            ),
            default=0,
        )
        self.backlog = (
            sum(not e.admitted for e in view.episodes) >= evaluator.config.batch_size
        )
        self.preparations += 1
        # Original search orchestration expects two arguments for its candidate
        # generators. Neither generator below reads scores or computes costs.
        return (), ()

    @staticmethod
    @lru_cache(maxsize=16)
    def grouped(layouts, role):
        result = defaultdict(lambda: defaultdict(list))
        for c in layouts:
            if role == "actor":
                gpu, topology = c.actor_gpus, c.actor_parallel
            else:
                gpu = c.rollout_gpus
                topology = (c.rollout_parallel, c.sglang_dp_size)
            result[gpu][topology].append(c)
        groups = {}
        for gpu, values in result.items():
            rows = list(values.values())
            if role == "actor":
                rows.sort(
                    key=lambda row: (
                        row[0].actor_pp,
                        row[0].actor_cp,
                        row[0].actor_tp,
                        row[0].order_key,
                    )
                )
            else:
                rows.sort(
                    key=lambda row: (
                        row[0].rollout_pp,
                        row[0].rollout_tp,
                        row[0].sglang_dp_size,
                        row[0].order_key,
                    )
                )
            groups[gpu] = tuple(tuple(row) for row in rows)
        return groups

    def pick_actor(self, rows, probe):
        valid = [
            c
            for c in rows
            if c.actor_max_tokens_per_mb is None
            or c.actor_max_tokens_per_mb >= self.longest
        ]
        if not valid:
            return None
        caps = sorted(
            {
                c.actor_max_tokens_per_mb
                for c in valid
                if c.actor_max_tokens_per_mb is not None
            }
        )
        target_cap = caps[0] if caps and probe % 3 == 2 else None
        target_n_mbs = (1, 2, 4, 8)[probe % 4]
        return min(
            valid,
            key=lambda c: (
                c.actor_pad_to_maximum,
                c.actor_max_tokens_per_mb != target_cap,
                abs(c.actor_n_mbs - target_n_mbs),
                c.order_key,
            ),
        )

    @staticmethod
    def pick_rollout(rows, probe):
        workflow, running, prefill = (
            (128, 64, 32768),
            (64, 32, 16384),
            (32, 16, 8192),
            (16, 8, 8192),
        )[probe % 4]
        return min(
            rows,
            key=lambda c: (
                abs(c.rollout_max_concurrent_rollouts - workflow),
                abs(c.sglang_max_running_requests - running),
                abs(c.sglang_max_prefill_tokens - prefill),
                c.order_key,
            ),
        )

    def valid(self, layout, budget):
        if layout.actor_gpus + layout.rollout_gpus > budget:
            return False
        try:
            self.costs.validate_layout(layout)
        except ValueError:
            return False
        self.validated += 1
        return True

    def diverse(self, grid, current, actors, rollouts, *_unused):
        a = self.grouped(tuple(actors), "actor")
        r = self.grouped(tuple(rollouts), "rollout")
        budget = grid["gpu_budget"]
        cells = defaultdict(list)
        for ag in a:
            for rg in r:
                if ag + rg <= budget:
                    cell = (min(3, ag * 4 // budget), min(3, rg * 4 // budget))
                    cells[cell].append((ag, rg))
        for pairs in cells.values():
            pairs.sort(
                key=lambda pair: (
                    -sum(pair),
                    -pair[0] if self.backlog else -pair[1],
                    abs(pair[0] - current.actor_gpus)
                    + abs(pair[1] - current.rollout_gpus),
                )
            )
        order = sorted(
            cells,
            key=lambda cell: (
                -cell[0] if self.backlog else -min(cell),
                -sum(cell),
                cell,
            ),
        )
        seen = set()
        # A bounded round-robin over allocation cells; each round also varies
        # topology and packing/engine settings. No Actor/Rollout timing calls.
        for probe in range(128):
            for cell in order:
                pairs = cells[cell]
                ag, rg = pairs[(probe // 4) % len(pairs)]
                actor_rows = a[ag][probe % len(a[ag])]
                rollout_rows = r[rg][(probe // 2) % len(r[rg])]
                actor = self.pick_actor(actor_rows, probe)
                if actor is None:
                    continue
                rollout = self.pick_rollout(rollout_rows, probe)
                layout = actor.with_rollout(rollout)
                effective = self.costs.resolve(layout)
                if effective not in seen and self.valid(layout, budget):
                    seen.add(effective)
                    yield layout

    def ordered(self, grid, current, actors, rollouts, *_unused):
        yield current
        neighbors = []
        for role, options in (("actor", actors), ("rollout", rollouts)):
            for option in options:
                c = (
                    option.with_rollout(current)
                    if role == "actor"
                    else current.with_rollout(option)
                )
                if distance(c, current) != 1:
                    continue
                if (
                    role == "actor"
                    and c.actor_max_tokens_per_mb is not None
                    and c.actor_max_tokens_per_mb < self.longest
                ):
                    continue
                if self.valid(c, grid["gpu_budget"]):
                    neighbors.append(c)
        neighbors.sort(
            key=lambda c: (
                -(c.actor_gpus + c.rollout_gpus)
                if self.backlog
                else c.actor_gpus + c.rollout_gpus,
                c.order_key,
            )
        )
        yield from neighbors
        yield from self.diverse(grid, current, actors, rollouts)

    def statistics(self):
        return dict(
            cost_scoring_calls=0,
            preparations=self.preparations,
            static_layout_checks=self.validated,
            observed_longest_sequence=self.longest,
            admission_backlog=self.backlog,
            grid=self._grids.cache_info()._asdict(),
        )
