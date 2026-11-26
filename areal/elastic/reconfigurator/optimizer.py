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
        reconfiguration: ReconfigurationModel | None = None,
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
        scale_in_only = (
            signal.actor_action != "scale_out" and signal.rollout_action != "scale_out"
        )

        split, lock_actor, lock_rollout = self._split(current, signal)
        if scale_in_only and sum(split) >= current.gpus:
            return Decision(
                current,
                current,
                split,
                0,
                0,
                "Scale-in does not reduce total GPUs; keeping current layout",
            )

        @cache
        def actor_capacity(role: RoleLayout):
            if role == current.actor:
                return signal.actor_capacity
            return self.evaluator.actor_capacity(role, pattern, publication)

        @cache
        def rollout_slo(role: RoleLayout, offered_rate: float):
            return self.evaluator.rollout_slo(role, pattern, offered_rate)

        def rollout_capacity(role: RoleLayout):
            if role == current.rollout:
                return current_rollout
            return probe_capacity(lambda load: rollout_slo(role, load), rate, target)

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
                actor_capacity.cache_info().currsize,
                rollout_slo.cache_info().currsize,
                reason,
            )

        for actor in actors:
            capacity = actor_capacity(actor)
            if capacity is None or capacity < rate:
                continue
            for layout in layouts((actor,), rollouts):
                score = rollout_slo(layout.rollout, rate)
                if score >= target:
                    if scale_in_only:
                        if layout.gpus >= current.gpus:
                            return result(
                                current,
                                "Scale-in does not reduce total GPUs; keeping current layout",
                            )
                        if self.reconfiguration is not None:
                            cutover = self.reconfiguration.predict(
                                current, layout, publication
                            )
                            saved = (current.gpus - layout.gpus) * self.horizon_seconds
                            if saved <= cutover.gpu_seconds:
                                return result(
                                    current,
                                    "Scale-in GPU savings do not pay back cutover cost; keeping current layout",
                                )
                    return result(layout, "Capacity and offered-task SLO meet demand")

        if scale_in_only:
            return result(
                current, "Scale-in target missed demand; keeping current layout"
            )

        @cache
        def rollout_lower(role: RoleLayout) -> float:
            return rollout_capacity(role).lower or 0.0

        feasible_actors = [role for role in actors if actor_capacity(role) is not None]
        best = max(
            layouts(feasible_actors, rollouts),
            key=lambda layout: (
                min(actor_capacity(layout.actor), rollout_lower(layout.rollout)),
                rollout_slo(layout.rollout, rate),
                layout == current,
            ),
            default=None,
        )
        if best is None:
            return result(None, "Selected GPU split has no placeable candidate")
        current_bottleneck = min(
            signal.actor_capacity or 0.0, current_rollout.lower or 0.0
        )
        best_bottleneck = min(actor_capacity(best.actor), rollout_lower(best.rollout))
        if best != current and best_bottleneck <= current_bottleneck:
            return result(
                current,
                "Selected GPU split does not improve bottleneck capacity; keeping current layout",
            )
        return result(best, "Selected GPU split misses demand; using its best capacity")
