# SPDX-License-Identifier: Apache-2.0

"""Choose a physical layout by independently scaling Actor and Rollout capacity."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import cache
from itertools import product
from typing import TYPE_CHECKING

from .trigger import ScaleSignal

if TYPE_CHECKING:
    from .evaluator import CapacityEvaluator, ReconfigurationModel
    from .evaluator.workload import ExecutionConfig


Value = bool | int | float | str | None
ATTENTION_DP_OPTIONS = frozenset(("sglang_dp_size", "sglang_enable_dp_attention"))


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
    from .evaluator.actor import MegatronCosts, actor_config
    from .evaluator.rollout import rollout_config
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
            rollout_config(models.rollout_template, models.rollout_model, config)
        except ValueError:
            continue
        rollouts.append(role)
    return replace(space, actors=tuple(actors), rollouts=tuple(rollouts))


@dataclass(frozen=True)
class Decision:
    source: Layout
    target: Layout | None
    split: tuple[int, int]
    actor_evaluations: int
    rollout_evaluations: int
    reason: str


class Optimizer:
    """Scale triggered role(s) proportionally, then search only that GPU split."""

    theta_up = 0.90

    def __init__(
        self,
        evaluator: CapacityEvaluator,
        gpu_capacity: int,
        place: Callable[[Layout], object],
        *,
        reconfiguration: ReconfigurationModel,
        horizon_seconds: float = 600.0,
    ):
        self.evaluator = evaluator
        self.place = place
        self.gpu_capacity = gpu_capacity
        self.reconfiguration = reconfiguration
        self.horizon_seconds = horizon_seconds
        self.space = supported_space(
            SearchSpace.for_pool(gpu_capacity, evaluator.batch_size), evaluator, (1,)
        )

    @property
    def min_actor_gpus(self) -> int:
        return min((r.gpus for r in self.space.actors), default=1)

    @property
    def min_rollout_gpus(self) -> int:
        return min((r.gpus for r in self.space.rollouts), default=1)

    def _project(
        self,
        target_actor: float,
        target_rollout: float,
        actor_counts: set[int],
        rollout_counts: set[int],
    ) -> tuple[int, int]:
        return min(
            (
                (a, r)
                for a, r in product(actor_counts, rollout_counts)
                if a + r <= self.gpu_capacity
            ),
            key=lambda pair: (
                round(
                    sum(
                        max(0.0, want - have)
                        for want, have in zip((target_actor, target_rollout), pair)
                    ),
                    12,
                ),
                abs(pair[0] - target_actor) + abs(pair[1] - target_rollout),
                pair,
            ),
        )

    def _split(
        self, current: Layout, signal: ScaleSignal
    ) -> tuple[tuple[int, int], bool, bool]:
        """Compute target (actor_gpus, rollout_gpus) and which roles stay locked."""
        rate = signal.rate
        actor_rate = signal.actor_capacity
        rollout_rate = signal.rollout_capacity.rate
        inv_target = 1.0 / self.theta_up

        requested_actor = (
            current.actor.gpus * (rate / actor_rate) * inv_target
            if actor_rate
            else float(self.gpu_capacity)
        )
        requested_rollout = (
            current.rollout.gpus * (rate / rollout_rate) * inv_target
            if rollout_rate
            else float(self.gpu_capacity)
        )

        actor_counts = {role.gpus for role in (*self.space.actors, current.actor)}
        rollout_counts = {role.gpus for role in (*self.space.rollouts, current.rollout)}

        # Single-role scale when only one role triggered and free pool GPUs suffice.
        if signal.actor_action != "hold" and signal.rollout_action == "hold":
            if requested_actor + current.rollout.gpus <= self.gpu_capacity:
                split = self._project(
                    requested_actor,
                    float(current.rollout.gpus),
                    actor_counts,
                    {current.rollout.gpus},
                )
                return split, False, True
        elif signal.rollout_action != "hold" and signal.actor_action == "hold":
            if current.actor.gpus + requested_rollout <= self.gpu_capacity:
                split = self._project(
                    float(current.actor.gpus),
                    requested_rollout,
                    {current.actor.gpus},
                    rollout_counts,
                )
                return split, True, False

        # Joint proportional split (both roles triggered or pool full -> borrow GPUs).
        scale = min(1.0, self.gpu_capacity / (requested_actor + requested_rollout))
        split = self._project(
            requested_actor * scale,
            requested_rollout * scale,
            actor_counts,
            rollout_counts,
        )
        return split, False, False

    def decide(self, current: Layout, signal: ScaleSignal) -> Decision:
        from .evaluator.rollout import probe_capacity

        rate, pattern = signal.rate, signal.pattern
        target = self.evaluator.slo.minimum_request_fraction
        publication = signal.publication
        current_rollout = signal.rollout_capacity
        split, lock_actor, lock_rollout = self._split(current, signal)

        actor_evaluations = rollout_evaluations = 0

        @cache
        def rollout_limit(role: RoleLayout):
            nonlocal rollout_evaluations
            if role == current.rollout:
                return current_rollout.lower
            estimate = probe_capacity(
                lambda load: self.evaluator.rollout_slo(role, pattern, load),
                rate,
                target,
                require_observed_rate=True,
            )
            rollout_evaluations += len(estimate.probes)
            return estimate.lower

        inherited = self.space.inherited_attention_dp

        def roles(catalogue, active, gpus):
            original = coordinates(active, inherited)

            def rank(role):
                shape = coordinates(role, inherited)
                return (
                    role != active,
                    tuple(shape[1:]) != tuple(original[1:]),
                    sum(abs(a - b) for a, b in zip(shape, original)),
                    role.key,
                )

            return sorted(
                {role for role in (*catalogue, active) if role.gpus == gpus},
                key=rank,
            )

        actors = (
            [current.actor]
            if lock_actor
            else roles(self.space.actors, current.actor, split[0])
        )
        rollouts = (
            [current.rollout]
            if lock_rollout
            else roles(self.space.rollouts, current.rollout, split[1])
        )

        def layouts(actor_roles, rollout_roles):
            for actor, rollout in product(actor_roles, rollout_roles):
                layout = Layout(actor, rollout)
                try:
                    self.place(layout)
                except ValueError:
                    continue
                yield layout

        def result(layout: Layout | None, reason: str) -> Decision:
            return Decision(
                current,
                layout,
                split,
                actor_evaluations,
                rollout_evaluations,
                reason,
            )

        horizon = self.horizon_seconds
        demand = signal.outstanding_episodes + rate * horizon

        current_capacity = min(
            signal.actor_capacity or 0.0, current_rollout.lower or 0.0
        )
        current_tasks = min(demand, current_capacity * horizon)
        current_score = (current_tasks, -current.gpus * horizon)

        def window_score(layout: Layout, capacity: float) -> tuple[float, float]:
            cutover = self.reconfiguration.predict(current, layout, publication)
            active_seconds = max(0.0, horizon - cutover.seconds)
            tasks = min(demand, capacity * active_seconds)
            gpu_seconds = cutover.gpu_seconds + layout.gpus * active_seconds
            return tasks, -gpu_seconds

        for actor in actors:
            capacity = signal.actor_capacity
            if actor != current.actor:
                capacity = self.evaluator.actor_capacity(actor, pattern, publication)
                actor_evaluations += 1
            if capacity is None or capacity < rate:
                continue
            for layout in layouts((actor,), rollouts):
                limit = rollout_limit(layout.rollout)
                if limit is None or limit < rate:
                    continue
                if window_score(layout, min(capacity, limit)) > current_score:
                    return result(
                        layout,
                        "Capacity, offered-task SLO and cutover payback meet requirements",
                    )

        return result(current, "No feasible candidate improves the window score")
