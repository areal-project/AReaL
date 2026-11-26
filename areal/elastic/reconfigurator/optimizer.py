# SPDX-License-Identifier: Apache-2.0

"""Budgeted hierarchical search over physical Actor and Rollout layouts."""

from __future__ import annotations

from bisect import bisect_right
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

from .trigger import StableWorkloadTrigger, WorkloadChange

if TYPE_CHECKING:
    from .evaluator import Observation, ReplayEvaluator
    from .evaluator.rollout.capacity import RolloutCapacity
    from .evaluator.workload import ExecutionConfig, WorkloadPattern


Value = bool | int | float | str | None
ATTENTION_DP_OPTIONS = frozenset(("sglang_dp_size", "sglang_enable_dp_attention"))
DEFAULT_EVALUATION_BUDGET = 32


def _rank_products(
    axes: tuple[Sequence[int], ...], limit: int
) -> Iterator[tuple[tuple[int, ...], int]]:
    """Enumerate physical meshes without expanding over-capacity prefixes."""
    if not axes:
        yield (), 1
        return
    degrees = axes[0]
    if isinstance(degrees, range) and degrees.step > 0:
        degrees = degrees[: bisect_right(degrees, limit)]
    for degree in degrees:
        if degree <= limit:
            for suffix, size in _rank_products(axes[1:], limit // degree):
                yield (degree, *suffix), degree * size


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
            physical = tuple(axes.get(name, (1,)) for name in names[:4])
            experts = tuple(product(*(axes.get(name, (1,)) for name in names[4:])))
            meshes = (
                (ranks + expert, gpus)
                for ranks, gpus in _rank_products(physical, self.gpu_capacity - 1)
                for expert in experts
            )
            for values, gpus in meshes:
                if role == "actor":
                    dp, tp, pp, cp, ep, etp = values
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
                    backend = f"sglang:d{replicas}t{tp}p{pp}"
                for settings in product(*options.values()):
                    rows.append(
                        RoleLayout(backend, gpus, tuple(sorted(zip(options, settings))))
                    )
            result.append(tuple(dict.fromkeys(rows)))
        return result[0], result[1]

    @classmethod
    def for_pool(cls, gpu_capacity: int, batch_size: int):
        """Enumerate integer degrees within the pool for Megatron and SGLang."""
        counts = range(1, gpu_capacity)
        return cls(
            gpu_capacity,
            batch_size,
            dict(
                dp=tuple(n for n in counts if batch_size % n == 0),
                tp=counts,
                pp=counts,
                cp=counts,
            ),
            dict(replicas=counts, tp=counts, pp=counts),
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
    from .evaluator.actor.cost_model import MegatronCosts, actor_config
    from .evaluator.rollout.cost_model import rollout_config
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
            actor_config(models.actor_template, config), models.actor_model.model
        ):
            actors.append(role)
    rollouts = []
    for role in space.rollouts:
        config = replace(template, rollout_backend=role.backend, **dict(role.options))
        try:
            rollout_config(
                models.rollout_template, models.rollout_model, config
            )  # model shape and weight/KV capacity
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
    actor_capacity: float | None = None
    admission_failures: int = 0
    rollout_queue_failures: int = 0
    rollout_decode_failures: int = 0
    unfinished_episodes: int = 0


class Evaluator(Protocol):
    def prepare(self, observation: Any, *, pattern: WorkloadPattern) -> Any: ...

    def evaluate(self, context: Any, layout: Layout) -> Prediction: ...

    def estimate_rollout_capacity(
        self,
        context: Any,
        layout: Layout,
        pattern: WorkloadPattern,
        rate: float,
        target: float,
    ) -> RolloutCapacity: ...

    def estimate_actor_capacity(self, context: Any, layout: Layout) -> float | None: ...


@dataclass(frozen=True)
class Decision:
    source: Layout
    target: Layout
    predictions: tuple[Prediction, ...]
    reason: str
    workload_change: tuple[str, ...] = ()
    rollout_capacity_probe: RolloutCapacity | None = None
    within_split_evaluations: int = 0


@dataclass(frozen=True)
class Application:
    source: Layout
    target: Layout
    completed_at: float
    error: BaseException | None = None


def _usable(p: Prediction) -> bool:
    return p.supported and p.request_fraction is not None and p.gpu_seconds is not None


def _meets(p: Prediction, target: float) -> bool:
    return p.request_fraction + 1e-12 >= target


def select(
    predictions: Sequence[Prediction], current: Layout, target: float
) -> Prediction | None:
    """Minimize GPU seconds among supported layouts that meet the serving SLO."""
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
        max_evaluations=DEFAULT_EVALUATION_BUDGET,
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
    actors: tuple[RoleLayout, ...]
    rollouts: tuple[RoleLayout, ...]

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
        max_evaluations: int = DEFAULT_EVALUATION_BUDGET,
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
        self.workload_trigger = StableWorkloadTrigger(
            getattr(evaluator, "batch_size", 128)
        )
        self._observed_layout: Layout | None = None
        self._proposed_pattern: tuple[Layout, WorkloadChange] | None = None
        self.inherited_dp = space.inherited_attention_dp

    @classmethod
    def from_grid(
        cls,
        grid: SearchSpace,
        evaluator: ReplayEvaluator,
        *,
        attention_dp_sizes=(1,),
        max_evaluations=DEFAULT_EVALUATION_BUDGET,
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
        from .evaluator import LoadedProfiles, ReplayEvaluator
        from .evaluator.actor.cost_model import actor_config

        evaluator = self.evaluator
        if not isinstance(evaluator, ReplayEvaluator) or not isinstance(
            evaluator.models, LoadedProfiles
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
                model = evaluator.models.actor_model
                execution = actor_config(evaluator.models.actor_template, config)
                peak = model.estimate_batch(execution, batch).peak_gib
                capacity = model.hardware.memory_gb * 1e9 / 2**30
            except ValueError:
                continue  # Replay also marks this mandatory batch unsupported.
            if peak <= capacity * (1 - evaluator.reserve):
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
                        tuple(item.role for item in actors[a]),
                        tuple(item.role for item in rollouts[r]),
                    )
                )
        return splits

    def decide(
        self,
        observation: Observation,
        current: Layout,
        *,
        confirmed: WorkloadChange | None = None,
    ) -> Decision:
        """A caller supplying a confirmed change owns its trigger's observation clock."""
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
        if confirmed is None and current != self._observed_layout:
            proposal = self._proposed_pattern
            self.workload_trigger.bind(
                proposal[1] if proposal and proposal[0] == current else None
            )
            self._observed_layout, self._proposed_pattern = current, None
        change = (
            confirmed if confirmed is not None else self.workload_trigger.observe(view)
        )
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
        best_feasible: Prediction | None = None

        def query(layout):
            nonlocal best_feasible
            row = self.evaluator.evaluate(context, layout)
            if row.layout != layout:
                raise ValueError("Evaluator returned a different layout")
            if row.gpu_seconds is not None and (
                not isfinite(row.gpu_seconds) or row.gpu_seconds < horizon * layout.gpus
            ):
                raise ValueError(
                    "Prediction violates the native replay GPU-time lower bound"
                )
            if (
                _usable(row)
                and _meets(row, self.target)
                and (
                    best_feasible is None or row.gpu_seconds < best_feasible.gpu_seconds
                )
            ):
                best_feasible = row
            predictions.append(row)
            return row

        source = query(current)
        splits = self._splits(current, self._actors_for_forecast(forecast))

        probe = None
        rate = change.rate
        if best_feasible is None and splits and self.limit > 1 and rate > 0:
            probe = self.evaluator.estimate_rollout_capacity(
                context, current, change.pattern, rate, self.target
            )
        rollout_rate = probe.rate if probe is not None else None
        feedback = source
        tested = {current}
        refined = set()
        within_split_evaluations = 0

        def pressure(row):
            actor = (
                row.actor_capacity is not None
                and row.actor_capacity < rate
                and (row.admission_failures > 0 or row.actor_sustainable is False)
            )
            queue = row.rollout_queue_failures
            decode = row.rollout_decode_failures
            return actor, queue, decode

        def capacity_order(split):
            anchor = feedback.layout
            if rate and feedback.actor_capacity:
                need_actor = anchor.actor.gpus * rate / feedback.actor_capacity
            elif rate and source.actor_capacity:
                need_actor = current.actor.gpus * rate / source.actor_capacity
            else:
                need_actor = current.actor.gpus
            need_rollout = (
                current.rollout.gpus * rate / rollout_rate
                if rollout_rate is not None
                else current.rollout.gpus
            )
            a, r = split.actor_gpus, split.rollout_gpus
            actor_gap = max(0.0, 1 - a / need_actor)
            rollout_gap = max(0.0, 1 - r / need_rollout)
            actor, queue, decode = pressure(feedback)
            rollout = queue + decode > 0
            if actor and not rollout:
                counterproductive = (a <= anchor.actor.gpus, r < anchor.rollout.gpus)
            elif rollout and not actor:
                counterproductive = (r <= anchor.rollout.gpus, a < anchor.actor.gpus)
            else:
                counterproductive = (False, False)
            distance = abs(a - need_actor) + abs(r - need_rollout)
            return (
                split.queried > 0,
                max(actor_gap, rollout_gap) > 0,
                *counterproductive,
                max(actor_gap, rollout_gap),
                actor_gap + rollout_gap,
                distance,
                split.queried,
                a,
                r,
            )

        def parallel_alternative(split, row):
            actor, queue, decode = pressure(row)
            source_layout = row.layout
            if actor and not (queue or decode):
                candidates = [
                    Layout(role, source_layout.rollout)
                    for role in split.actors
                    if Layout(role, source_layout.rollout) not in tested
                ]
                estimates = [
                    (
                        self.evaluator.estimate_actor_capacity(context, candidate),
                        candidate,
                    )
                    for candidate in candidates[:4]
                ]
                return max(
                    (
                        item
                        for item in estimates
                        if item[0] and item[0] > row.actor_capacity
                    ),
                    key=lambda item: item[0],
                    default=(None, None),
                )[1]
            if not actor and (queue or decode):
                axis = 1 if decode > queue else 0  # TP for decode; replicas for queue.
                parallelism = coordinates(source_layout.rollout, self.inherited_dp)[
                    axis
                ]
                return next(
                    (
                        candidate
                        for role in split.rollouts
                        if coordinates(role, self.inherited_dp)[axis] > parallelism
                        and (candidate := Layout(source_layout.actor, role))
                        not in tested
                    ),
                    None,
                )
            return None

        while splits:
            alternative = None
            if best_feasible is not None:
                incumbent = best_feasible.layout
                # Try the closest smaller GPU count first. Same-size or larger
                # layouts can still win if they avoid enough switching cost.
                split = min(
                    (s for s in splits if horizon * s.gpus < best_feasible.gpu_seconds),
                    key=lambda s: (
                        s.gpus >= incumbent.gpus,
                        abs(s.gpus - incumbent.gpus),
                        s.queried,
                        abs(s.actor_gpus - incumbent.actor.gpus)
                        + abs(s.rollout_gpus - incumbent.rollout.gpus),
                        s.actor_gpus,
                        s.rollout_gpus,
                    ),
                    default=None,
                )
                if split is None:
                    reason = "Minimum cost certified by GPU-time bound"
                    break
            else:
                split = min(splits, key=capacity_order)
            if len(predictions) == self.limit:
                reason = "Query budget exhausted"
                break
            if best_feasible is None and feedback is not source:
                key = (feedback.layout.actor.gpus, feedback.layout.rollout.gpus)
                if key not in refined:
                    peer = next(
                        (s for s in splits if (s.actor_gpus, s.rollout_gpus) == key),
                        None,
                    )
                    if peer is not None:
                        alternative = parallel_alternative(peer, feedback)
                        if alternative is not None:
                            refined.add(key)
                            within_split_evaluations += 1
                            split = peer
            if alternative is None:
                alternative = next(
                    candidate for candidate in split.layouts if candidate not in tested
                )
            row = query(alternative)
            tested.add(alternative)
            if _usable(row):
                feedback = row
            split.queried += 1
            if split.queried == split.count:
                splits.remove(split)
        else:
            reason = "Catalogue exhausted"
        winner = select(predictions, current, self.target)
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
        if confirmed is None and winner is not None:
            if winner.layout == current:
                self.workload_trigger.bind(change)
                self._proposed_pattern = None
            else:
                self._proposed_pattern = (winner.layout, change)
        return Decision(
            current,
            winner.layout if winner else current,
            tuple(predictions),
            reason,
            change.reasons,
            probe,
            within_split_evaluations,
        )
