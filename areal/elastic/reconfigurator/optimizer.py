# SPDX-License-Identifier: Apache-2.0

"""Budgeted hierarchical search over physical Actor and Rollout layouts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from functools import cache
from heapq import merge
from itertools import product, repeat
from math import inf, isfinite
from time import monotonic
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from .trigger import WorkloadTrigger

if TYPE_CHECKING:
    from .evaluator import Observation, ReplayEvaluator
    from .evaluator.workload import ExecutionConfig, WorkloadPattern


Value = bool | int | float | str | None
ATTENTION_DP_OPTIONS = frozenset(("sglang_dp_size", "sglang_enable_dp_attention"))


@dataclass(frozen=True)
class RoleLayout:
    backend: str
    gpus: int
    options: tuple[tuple[str, Value], ...] = ()

    @property
    def key(self) -> tuple:
        return self.backend, self.gpus, repr(self.options)


@dataclass(frozen=True)
class Layout:
    actor: RoleLayout
    rollout: RoleLayout

    @property
    def gpus(self) -> int:
        return self.actor.gpus + self.rollout.gpus

    @property
    def key(self) -> tuple:
        return self.actor.key, self.rollout.key


@dataclass
class SearchSpace:
    gpu_capacity: int
    batch_size: int
    actor: Mapping[str, Sequence[int]]
    rollout: Mapping[str, Sequence[int]]
    parameters: Mapping[str, Sequence[Value]] = field(default_factory=dict)
    moe: bool = False

    def roles(self) -> tuple[tuple[RoleLayout, ...], tuple[RoleLayout, ...]]:
        """Expand role axes once; GPU/model support is not inferred from timing."""
        dimensions = (("dp", "tp", "pp", "cp", "ep", "etp"), ("replicas", "tp", "pp"))
        if self.gpu_capacity < 2 or self.batch_size < 1:
            raise ValueError("A positive batch and GPUs for both roles are required")
        result = []
        for role, axes, names in zip(
            ("actor", "rollout"), (self.actor, self.rollout), dimensions
        ):
            if set(axes) - set(names):
                raise ValueError(f"Unknown {role} parallel dimension")
            if any(
                not values or any(type(v) is not int or v < 1 for v in values)
                for values in axes.values()
            ):
                raise ValueError("Parallel dimensions require positive integer choices")
            options = {
                k: tuple(v)
                for k, v in self.parameters.items()
                if k.startswith("actor_") == (role == "actor")
            }
            rows = []
            for values in product(*(axes.get(name, (1,)) for name in names)):
                if role == "actor":
                    dp, tp, pp, cp, ep, etp = values
                    gpus = dp * tp * pp * cp
                    if (
                        self.batch_size % dp
                        or dp * tp * cp % (ep * etp)
                        or (not self.moe and (ep != 1 or etp != 1))
                    ):
                        continue
                    parallel = f"d{dp}t{tp}p{pp}c{cp}"
                    if ep != 1 or etp != 1:
                        parallel = f"(attn:{parallel}|ffn:d{dp * tp * cp // (ep * etp)}t{etp}p{pp}e{ep})"
                    backend = "megatron:" + parallel
                else:
                    replicas, tp, pp = values
                    gpus = replicas * tp * pp
                    backend = f"sglang:d{replicas}t{tp}p{pp}"
                if gpus >= self.gpu_capacity:
                    continue
                for settings in product(*options.values()):
                    rows.append(
                        RoleLayout(backend, gpus, tuple(sorted(zip(options, settings))))
                    )
            result.append(tuple(dict.fromkeys(rows)))
        return result[0], result[1]

    @classmethod
    def for_pool(cls, gpu_capacity: int, batch_size: int):
        """Default physical axes for a Megatron Actor and homogeneous SGLang replicas."""
        counts = range(1, gpu_capacity)
        tp = (1, 2, 4, 8)
        stages = (*tp, 16)
        return cls(
            gpu_capacity,
            batch_size,
            dict(dp=counts, tp=tp, pp=stages, cp=stages),
            dict(replicas=counts, tp=tp, pp=stages),
        )

    @classmethod
    def full_roles(cls, gpu_capacity, evaluator):
        grid = cls.for_pool(gpu_capacity, evaluator.batch_size)
        return supported_space(grid, evaluator, (1,)).roles()


@dataclass
class PhysicalSpace:
    gpu_capacity: int
    inherited_attention_dp: int
    actors: tuple[RoleLayout, ...]
    rollouts: tuple[RoleLayout, ...]

    def roles(self):
        return self.actors, self.rollouts


def physical_space(base: SearchSpace, template: ExecutionConfig, attention_dp_sizes):
    """Enumerate only the parallelism controls accepted by the Trainer API."""
    if any(value != 1 for dim in ("ep", "etp") for value in base.actor.get(dim, ())):
        raise ValueError("Expert parallel reconfiguration is not supported")
    if base.parameters.keys() - ATTENTION_DP_OPTIONS:
        raise ValueError(
            "Supply physical parallelism axes only; engine tuning remains fixed"
        )
    sizes = base.parameters.get("sglang_dp_size", attention_dp_sizes)
    if any(type(dp) is not int or dp < 1 for dp in sizes):
        raise ValueError("Attention DP must be a positive integer")
    enabled = base.parameters.get("sglang_enable_dp_attention", (False,))
    actors, rollouts = replace(base, parameters={}).roles()
    if 1 not in sizes or False not in enabled:
        rollouts = ()
    elif template.sglang_enable_dp_attention or template.sglang_dp_size != 1:
        # Disable inherited Attention DP for new targets, including TP > 1.
        off = (("sglang_dp_size", 1), ("sglang_enable_dp_attention", False))
        rollouts = tuple(replace(role, options=off) for role in rollouts)
    return PhysicalSpace(base.gpu_capacity, template.sglang_dp_size, actors, rollouts)


@cache
def coordinates(role, inherited_dp=1):
    """Expose physical layout dimensions; CP belongs to the Actor."""
    from .evaluator.workload import ExecutionConfig, parse_actor_parallel

    if role.backend.startswith("megatron:"):
        return parse_actor_parallel(role.backend)
    config = ExecutionConfig("megatron:d1", role.backend)
    return (
        *config.rollout_parallel,
        dict(role.options).get("sglang_dp_size", inherited_dp),
    )


def supported_space(grid, evaluator, attention_dp_sizes):
    from .evaluator.cost_model.actor.megatron import MegatronCosts
    from .evaluator.workload import ExecutionConfig

    template = getattr(evaluator, "template", None)
    if template is None:
        template = ExecutionConfig("megatron:d1", "sglang:d1")
    space = physical_space(grid, template, attention_dp_sizes)
    models = getattr(evaluator, "models", None)
    if models is None:
        return space
    actors = []
    for role in space.actors:
        config = replace(template, actor_backend=role.backend)
        if MegatronCosts.mesh_is_valid(
            models.actor_config(config), models.actor_model.model
        ):
            actors.append(role)
    rollouts = []
    for role in space.rollouts:
        config = replace(template, rollout_backend=role.backend, **dict(role.options))
        try:
            models.rollout_config(config)  # model shape and weight/KV capacity
        except ValueError:
            continue
        rollouts.append(role)
    return replace(space, actors=tuple(actors), rollouts=tuple(rollouts))


@dataclass(frozen=True)
class Prediction:
    layout: Layout
    request_fraction: float | None
    gpu_seconds: float | None
    actor_sustainable: bool | None
    actor_deficit: float | None = None
    supported: bool = True
    reason: str = ""
    completed_updates: int | None = None
    pending_requests: int | None = None
    completed_requests: int | None = None


class Evaluator(Protocol):
    def prepare(self, observation: Any, *, pattern: WorkloadPattern) -> Any: ...

    def evaluate(self, context: Any, layout: Layout) -> Prediction: ...


@dataclass(frozen=True)
class Decision:
    source: Layout
    target: Layout
    predictions: tuple[Prediction, ...]
    reason: str
    workload_change: tuple[str, ...] = ()


@dataclass(frozen=True)
class Application:
    source: Layout
    target: Layout
    completed_at: float
    error: BaseException | None = None


def _usable(p: Prediction) -> bool:
    return p.supported and p.request_fraction is not None and p.gpu_seconds is not None


def _meets(p: Prediction, target: float) -> bool:
    return p.actor_sustainable is True and p.request_fraction + 1e-12 >= target


def select(
    predictions: Sequence[Prediction], current: Layout, target: float
) -> Prediction | None:
    """Minimize GPU seconds subject to serving SLO and sustainable training."""
    return min(
        (p for p in predictions if _usable(p) and _meets(p, target)),
        key=lambda p: (p.gpu_seconds, p.layout != current, p.layout.key),
        default=None,
    )


class BaseOptimizer:
    """Shared decision budget and submission lifecycle for layout policies."""

    def __init__(
        self,
        space,
        evaluator,
        *,
        minimum_request_fraction,
        max_evaluations=64,
        clock=monotonic,
    ):
        if (
            not 0 < minimum_request_fraction <= 1
            or type(max_evaluations) is not int
            or max_evaluations < 1
        ):
            raise ValueError(
                "A valid SLO target and positive search allowance are required"
            )
        self.actors, self.rollouts = space.roles()
        self._actors, self._rollouts = set(self.actors), set(self.rollouts)
        self.gpu_capacity = space.gpu_capacity
        self.evaluator, self.target, self.limit = (
            evaluator,
            minimum_request_fraction,
            max_evaluations,
        )
        self.clock = clock
        self.pending: Future | None = None
        self.last_application: Application | None = None
        self.last_switch: float | None = None

    def request(
        self, decision: Decision, current: Layout, submit: Callable[[Layout], Future]
    ) -> Future | None:
        """The caller maps layouts to Trainer arguments; no worker control lives here.

        The source is checked at submission. Execution-time source validation, if
        required, belongs to the submission adapter; a Future does not provide it.
        """
        if self.pending is not None:
            raise RuntimeError("A reconfiguration is already pending")
        if decision.source != current:
            raise ValueError("The source layout changed after the decision")
        if decision.target == current:
            return None
        future = submit(decision.target)
        self.pending = future

        def completed(done):
            error = None
            try:
                done.result()
            except BaseException as exc:
                error = exc
            at = self.clock()
            self.last_application = Application(current, decision.target, at, error)
            if error is None:
                self.last_switch = at
            self.pending = None

        future.add_done_callback(completed)
        return future


class _RankedRole(NamedTuple):
    changed: bool
    distance: int
    role: RoleLayout


def _rank_roles(roles, source, inherited_dp):
    origin = coordinates(source, inherited_dp)
    by_gpus = defaultdict(list)
    for role in roles:
        distance = sum(
            a != b for a, b in zip(origin, coordinates(role, inherited_dp), strict=True)
        )
        by_gpus[role.gpus].append(_RankedRole(role != source, distance, role))
    for group in by_gpus.values():
        group.sort(key=lambda item: (item.changed, item.distance, item.role.key))
    return by_gpus


def _joint_layouts(actors, rollouts, current):
    # Each fixed-Actor stream already has the desired ordering. Merge without
    # materializing the product of all Actor and rollout configurations.
    streams = (zip(repeat(actor), rollouts) for actor in actors)

    def order(pair):
        a, r = pair
        return a.changed + r.changed, a.distance + r.distance, a.role.key, r.role.key

    for a, r in merge(*streams, key=order):
        layout = Layout(a.role, r.role)
        if layout != current:
            yield layout


@dataclass
class _Split:
    actor_gpus: int
    rollout_gpus: int
    layouts: Iterator[Layout]
    count: int
    queried: int

    @property
    def gpus(self):
        return self.actor_gpus + self.rollout_gpus


class Optimizer(BaseOptimizer):
    """Search complete layouts by GPU split with a bounded number of replay queries."""

    def __init__(
        self,
        space: SearchSpace | PhysicalSpace | None = None,
        evaluator: Evaluator | None = None,
        *,
        gpu_capacity: int | None = None,
        minimum_request_fraction: float,
        max_evaluations: int = 64,
        clock: Callable[[], float] = monotonic,
        attention_dp_sizes: Sequence[int] = (1,),
    ):
        if evaluator is None or (space is None) == (gpu_capacity is None):
            raise ValueError(
                "Supply an evaluator and either GPU capacity or an explicit search space"
            )
        if space is None:
            space = SearchSpace.for_pool(gpu_capacity, evaluator.batch_size)
        if isinstance(space, SearchSpace):
            space = supported_space(space, evaluator, attention_dp_sizes)
        else:
            if any(role.options for role in space.actors) or any(
                key not in ATTENTION_DP_OPTIONS
                for role in space.rollouts
                for key, _ in role.options
            ):
                raise ValueError("Set fixed runtime settings on the evaluator template")
            template = getattr(evaluator, "template", None)
            inherited_size = getattr(
                template, "sglang_dp_size", space.inherited_attention_dp
            )
            inherited_enabled = getattr(
                template, "sglang_enable_dp_attention", inherited_size > 1
            )

            def attention_dp_off(role):
                options = dict(role.options)
                return options.get("sglang_dp_size", inherited_size) == 1 and not (
                    options.get("sglang_enable_dp_attention", inherited_enabled)
                )

            space = replace(
                space, rollouts=tuple(r for r in space.rollouts if attention_dp_off(r))
            )
        super().__init__(
            space,
            evaluator,
            minimum_request_fraction=minimum_request_fraction,
            max_evaluations=max_evaluations,
            clock=clock,
        )
        self.workload_trigger = WorkloadTrigger()
        self.inherited_dp = space.inherited_attention_dp

    @classmethod
    def from_grid(
        cls,
        grid: SearchSpace,
        evaluator: ReplayEvaluator,
        *,
        attention_dp_sizes=(1,),
        max_evaluations=64,
    ):
        """Normal entry point. Build a physical catalogue from the native profiles once."""
        return cls(
            grid,
            evaluator,
            minimum_request_fraction=evaluator.slo.minimum_request_fraction,
            max_evaluations=max_evaluations,
            attention_dp_sizes=attention_dp_sizes,
        )

    def _actors_for_forecast(self, forecast):
        from .evaluator import ReplayEvaluator
        from .evaluator.cost_model.models import AnalyticalCostModels

        evaluator = self.evaluator
        if not isinstance(evaluator, ReplayEvaluator) or not isinstance(
            evaluator.models, AnalyticalCostModels
        ):
            return self.actors
        # Replay checks this first batch for every layout, before scheduling any
        # rollout. The analytical Actor memory estimate uses only Actor settings.
        episodes = forecast.episodes
        if not episodes:
            return self.actors  # Only the carried backlog remains.
        batch = tuple(
            episodes[i % len(episodes)].turns for i in range(evaluator.batch_size)
        )
        actors = []
        for role in self.actors:
            config = replace(
                evaluator.template, actor_backend=role.backend, **dict(role.options)
            )
            try:
                peak, capacity, oom = evaluator.models.actor_memory(config, batch)
            except ValueError:
                continue  # Replay also marks this mandatory batch unsupported.
            if not (oom or peak > capacity * (1 - evaluator.reserve)):
                actors.append(role)
        return actors

    def _splits(self, current, eligible_actors):
        actors = _rank_roles(eligible_actors, current.actor, self.inherited_dp)
        rollouts = _rank_roles(self.rollouts, current.rollout, self.inherited_dp)
        source_in_catalogue = (
            current.actor in eligible_actors and current.rollout in self._rollouts
        )
        splits = []
        for a, r in product(actors, rollouts):
            if a + r > self.gpu_capacity:
                continue
            queried = int(
                source_in_catalogue
                and (a, r) == (current.actor.gpus, current.rollout.gpus)
            )
            count = len(actors[a]) * len(rollouts[r])
            if queried < count:
                splits.append(
                    _Split(
                        a,
                        r,
                        _joint_layouts(actors[a], rollouts[r], current),
                        count,
                        queried,
                    )
                )
        return splits

    def decide(self, observation: Observation, current: Layout) -> Decision:
        if self.pending is not None:
            return Decision(current, current, (), "Adjustment pending")
        if current != observation.current or current.gpus > self.gpu_capacity:
            raise ValueError(
                "Current layout must match the observation and fit the pool"
            )
        if (
            current.actor.options
            or dict(current.rollout.options).keys() - ATTENTION_DP_OPTIONS
        ):
            raise ValueError("Set fixed runtime settings on the evaluator template")
        view = observation.view
        if view.busy or view.training_version != view.provider_version:
            return Decision(
                current, current, (), "Waiting for a published update boundary"
            )
        change = self.workload_trigger.observe(view)
        if change is None:
            return Decision(current, current, (), "No stable workload change")
        context = self.evaluator.prepare(observation, pattern=change.pattern)
        forecast = context[1]
        if forecast is None or (
            not forecast.episodes
            and not (context[0].backlog and context[0].backlog.shapes)
        ):
            return Decision(
                current,
                current,
                (),
                "No workload or publication in the forecast window",
            )
        horizon = forecast.input_closes_at - context[0].view.now
        if not isfinite(horizon) or horizon <= 0:
            raise ValueError("Expected a positive, finite forecast horizon")

        predictions = []
        best_cost = inf

        def query(layout):
            nonlocal best_cost
            row = self.evaluator.evaluate(context, layout)
            if row.layout != layout:
                raise ValueError("Evaluator returned a different layout")
            if row.gpu_seconds is not None and (
                not isfinite(row.gpu_seconds) or row.gpu_seconds < horizon * layout.gpus
            ):
                raise ValueError(
                    "Prediction violates the native replay GPU-time lower bound"
                )
            if _usable(row) and _meets(row, self.target):
                best_cost = min(best_cost, row.gpu_seconds)
            predictions.append(row)
            return row

        source = query(current)
        splits = self._splits(current, self._actors_for_forecast(forecast))

        def discovery_order(split):
            preferred = True
            if source.supported and source.actor_sustainable is False:
                preferred = split.actor_gpus > current.actor.gpus
            elif (
                source.supported
                and source.actor_sustainable is True
                and source.request_fraction is not None
                and source.request_fraction < self.target
            ):
                preferred = split.rollout_gpus > current.rollout.gpus
            distance = abs(split.actor_gpus - current.actor.gpus) + abs(
                split.rollout_gpus - current.rollout.gpus
            )
            return (
                split.queried,
                not preferred,
                distance,
                split.actor_gpus,
                split.rollout_gpus,
            )

        while splits:
            if best_cost < inf:
                split = min(
                    splits,
                    key=lambda s: (s.gpus, s.queried, s.actor_gpus, s.rollout_gpus),
                )
                if horizon * split.gpus >= best_cost:
                    reason = "Minimum cost certified by GPU-time bound"
                    break
            else:
                split = min(splits, key=discovery_order)
            if len(predictions) == self.limit:
                reason = "Query budget exhausted"
                break
            query(next(split.layouts))
            split.queried += 1
            if split.queried == split.count:
                splits.remove(split)
        else:
            reason = "Catalogue exhausted"
        winner = select(predictions, current, self.target)
        self.workload_trigger.searched(change)
        if winner is None:
            reason = (
                f"No feasible candidate among {len(predictions)} evaluated ({reason})"
            )
            usable = (p for p in predictions if _usable(p))
            winner = min(
                usable,
                key=lambda p: (
                    p.actor_sustainable is not True,
                    0
                    if p.actor_sustainable
                    else p.actor_deficit
                    if p.actor_deficit is not None
                    else inf,
                    -p.request_fraction,
                    p.pending_requests if p.pending_requests is not None else inf,
                    p.gpu_seconds,
                    p.layout != current,
                    p.layout.key,
                ),
                default=None,
            )
            if winner is not None:
                reason += "; selected best available"
        return Decision(
            current,
            winner.layout if winner else current,
            tuple(predictions),
            reason,
            change.reasons,
        )
