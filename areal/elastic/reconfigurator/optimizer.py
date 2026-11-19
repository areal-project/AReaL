# SPDX-License-Identifier: Apache-2.0

"""Automatic layout search with an injected evaluator and submission function.

The evaluator owns workload prediction and model support. The caller owns
observations, triggering and the native Trainer. This module starts no services.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from itertools import product
from math import inf
from time import monotonic
from typing import Any, Literal, Protocol

Value = bool | int | float | str | None


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
    def full_roles(cls, gpu_capacity, evaluator):
        """Reference's Megatron/SGLang grid. Expand each role, never the joint product."""
        models = evaluator.models
        base = models.resolve(evaluator.template)
        if base.rollout_max_concurrent_rollouts is None:
            base = replace(base, rollout_max_concurrent_rollouts=evaluator.batch_size)
        models.validate_layout(base)
        moe = models.actor_model.model.mlp_type == "moe"
        counts = range(1, gpu_capacity)
        tp = (1, 2, 4, 8)
        stages = (*tp, 16)
        space = cls(
            gpu_capacity,
            evaluator.batch_size,
            dict(
                dp=counts,
                tp=tp,
                pp=stages,
                cp=stages,
                # Expert-parallel search awaits reconfiguration validation.
                ep=(1,),
                etp=(1,),
            ),
            dict(replicas=counts, tp=tp, pp=stages),
            moe=moe,
        )
        actors, rollouts = space.roles()

        # Omit inherited values from the identity so the current configuration
        # and its copy in the generated space compare equal.
        def options(axes):
            names = sorted(axes)
            choices = [
                tuple(dict.fromkeys((*axes[n], getattr(base, n)))) for n in names
            ]
            return [
                tuple((n, v) for n, v in zip(names, values) if v != getattr(base, n))
                for values in product(*choices)
            ]

        actor_options = options(
            dict(
                actor_n_mbs=(1, 2, 4, 8),
                actor_max_tokens_per_mb=(8192, 16384, 32768, 65536),
                actor_pad_to_maximum=(False, True),
            )
        )
        concurrency = evaluator.concurrency
        rollout_options = options(
            dict(
                rollout_max_concurrent_rollouts=tuple(
                    max(1, concurrency // n) for n in (8, 4, 2, 1)
                ),
                sglang_max_running_requests=(8, 16, 32, 64),
                sglang_max_prefill_tokens=(8192, 16384, 32768),
            )
        )
        valid_actors, valid_rollouts = [], []
        for role in actors:
            cfg = replace(base, actor_backend=role.backend)
            if cfg.actor_pp > models.actor_model.model.layers:
                continue
            try:
                models.validate_layout(cfg)
            except ValueError:
                continue
            valid_actors.extend(replace(role, options=o) for o in actor_options)
        for role in rollouts:
            cfg = replace(
                base,
                rollout_backend=role.backend,
                sglang_dp_size=1,
                sglang_enable_dp_attention=False,
            )
            divisors = [d for d in tp if cfg.rollout_tp % d == 0]
            for dp in divisors:
                internal = dict(
                    sglang_dp_size=dp,
                    sglang_enable_dp_attention=dp > 1,
                )
                # Check the mesh with at least one running slot per DP group.
                try:
                    models.validate_layout(
                        replace(
                            cfg,
                            **internal,
                            sglang_max_running_requests=max(
                                base.sglang_max_running_requests, dp
                            ),
                        )
                    )
                except ValueError:
                    continue
                changed = [(k, v) for k, v in internal.items() if v != getattr(base, k)]
                for o in rollout_options:
                    values = dict(o)
                    if (
                        values.get(
                            "sglang_max_running_requests",
                            base.sglang_max_running_requests,
                        )
                        < dp
                    ):
                        continue
                    valid_rollouts.append(
                        replace(role, options=tuple(sorted((*changed, *o))))
                    )
        return tuple(valid_actors), tuple(valid_rollouts)


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
    def prepare(self, observation: Any) -> Any: ...

    def evaluate(self, context: Any, layout: Layout) -> Prediction: ...


@dataclass(frozen=True)
class Decision:
    source: Layout
    target: Layout
    predictions: tuple[Prediction, ...]
    reason: str


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


def _recovery_key(p: Prediction) -> tuple:
    deficit = (
        max(0.0, p.actor_deficit)
        if p.actor_deficit is not None
        else 0.0
        if p.actor_sustainable is True
        else inf
    )
    return (
        -(p.completed_updates or 0),
        p.pending_requests if p.pending_requests is not None else inf,
        -(p.completed_requests or 0),
        deficit,
        -p.request_fraction,
        p.gpu_seconds,
        p.layout.key,
    )


def select(
    predictions: Sequence[Prediction],
    current: Layout,
    target: float,
    *,
    trigger: str,
    recovering: bool = False,
) -> Prediction | None:
    """Choose the lowest-cost feasible layout, keeping the current layout on ties."""
    rows = [p for p in predictions if _usable(p)]
    feasible = [p for p in rows if _meets(p, target)]
    if feasible:
        return min(
            feasible,
            key=lambda p: (p.gpu_seconds, p.layout != current, p.layout.key),
        )
    if trigger == "slack":
        return next((p for p in rows if p.layout == current), None)
    if recovering and rows:
        best = max(p.request_fraction for p in rows)
        return min(
            (p for p in rows if p.request_fraction >= best - 0.02), key=_recovery_key
        )
    return min(
        rows,
        key=lambda p: (
            -p.request_fraction,
            _recovery_key(p)[3],
            p.gpu_seconds,
            p.layout != current,
            p.layout.key,
        ),
        default=None,
    )


class Optimizer:
    """One owner calls decide/request; Future callbacks record completion only.

    max_evaluations bounds policy computation, not training or Agent execution.
    No token, episode, training-update or runtime stop budget is introduced.
    """

    def __init__(
        self,
        space: SearchSpace | None = None,
        evaluator: Evaluator | None = None,
        *,
        gpu_capacity: int | None = None,
        minimum_request_fraction: float,
        max_evaluations: int = 64,
        clock: Callable[[], float] = monotonic,
    ):
        if (
            not 0 < minimum_request_fraction <= 1
            or type(max_evaluations) is not int
            or max_evaluations < 1
        ):
            raise ValueError(
                "A valid SLO target and positive search allowance are required"
            )
        if evaluator is None or (space is None) == (gpu_capacity is None):
            raise ValueError(
                "Supply an evaluator and either GPU capacity or an explicit search space"
            )
        if space is None:
            self.actors, self.rollouts = SearchSpace.full_roles(gpu_capacity, evaluator)
        else:
            self.actors, self.rollouts = space.roles()
            gpu_capacity = space.gpu_capacity
        self._actors, self._rollouts = set(self.actors), set(self.rollouts)
        self.gpu_capacity = gpu_capacity
        self.evaluator, self.target, self.limit = (
            evaluator,
            minimum_request_fraction,
            max_evaluations,
        )
        self.clock = clock
        self.history: deque[Layout] = deque(maxlen=7)
        self.pending: Future | None = None
        self.last_application: Application | None = None
        self.last_switch: float | None = None

    def _diverse(self, recovering: bool):
        """Visit each GPU allocation in turn, expanding its role combinations lazily."""
        actors, rollouts = defaultdict(list), defaultdict(list)
        for role in self.actors:
            actors[role.gpus].append(role)
        for role in self.rollouts:
            rollouts[role.gpus].append(role)
        allocations = [
            (a, r) for a in actors for r in rollouts if a + r <= self.gpu_capacity
        ]
        allocations.sort(
            key=lambda pair: (-pair[0] if recovering else -min(pair), -sum(pair), pair)
        )
        streams = deque(
            iter(
                product(
                    sorted(actors[a], key=lambda x: x.key),
                    sorted(rollouts[r], key=lambda x: x.key),
                )
            )
            for a, r in allocations
        )
        while streams:
            stream = streams.popleft()
            pair = next(stream, None)
            if pair is not None:
                yield Layout(*pair)
                streams.append(stream)

    @staticmethod
    def _distance(left: Layout, right: Layout) -> int:
        changes = 0
        for a, b in ((left.actor, right.actor), (left.rollout, right.rollout)):
            changes += a.backend != b.backend
            x, y = dict(a.options), dict(b.options)
            changes += sum(x.get(k) != y.get(k) for k in x.keys() | y.keys())
        return changes

    def decide(
        self,
        observation: Any,
        current: Layout,
        *,
        trigger: Literal["pressure", "slack", "none"],
        recovering: bool = False,
    ) -> Decision:
        if self.pending is not None or trigger == "none":
            return Decision(
                current,
                current,
                (),
                "Adjustment pending" if self.pending else "No trigger",
            )
        context = self.evaluator.prepare(observation)
        pool, seen, predictions = [], set(), []

        def add(layout):
            allowed = layout == current or (
                layout.actor in self._actors and layout.rollout in self._rollouts
            )
            if allowed and layout.gpus <= self.gpu_capacity and layout not in seen:
                seen.add(layout)
                pool.append(layout)

        def evaluate_until(stop):
            layouts = pool[len(predictions) : min(stop, len(pool), self.limit)]
            evaluate_many = getattr(self.evaluator, "evaluate_many", None)
            predictions.extend(
                evaluate_many(context, layouts)
                if evaluate_many is not None
                else (self.evaluator.evaluate(context, layout) for layout in layouts)
            )

        add(current)
        for previous in reversed(self.history):
            add(previous)
        for role, choices in (("actor", self.actors), ("rollout", self.rollouts)):
            neighbors = 0
            for choice in sorted(choices, key=lambda x: x.key):
                layout = (
                    Layout(choice, current.rollout)
                    if role == "actor"
                    else Layout(current.actor, choice)
                )
                if (
                    neighbors < 4
                    and layout not in seen
                    and self._distance(layout, current) == 1
                ):
                    before = len(pool)
                    add(layout)
                    neighbors += len(pool) > before
        diverse = self._diverse(recovering)
        initial = max(1, self.limit - 16)
        while len(pool) < initial:
            candidate = next(diverse, None)
            if candidate is None:
                break
            add(candidate)
        evaluate_until(initial)

        ranked = sorted(
            (p for p in predictions if _usable(p)),
            key=lambda p: (
                not _meets(p, self.target),
                -p.request_fraction,
                *_recovery_key(p),
            ),
        )
        actors = tuple(dict.fromkeys(p.layout.actor for p in ranked))[:4]
        rollouts = tuple(dict.fromkeys(p.layout.rollout for p in ranked))[:4]
        for a, r in product(actors, rollouts):
            if len(pool) < self.limit:
                add(Layout(a, r))
        while len(pool) < self.limit:
            candidate = next(diverse, None)
            if candidate is None:
                break
            add(candidate)
        evaluate_until(self.limit)
        best = select(
            predictions, current, self.target, trigger=trigger, recovering=recovering
        )
        if best is None:
            return Decision(
                current, current, tuple(predictions), "No supported prediction"
            )
        if best.layout in self.history:
            self.history.remove(best.layout)
        self.history.append(best.layout)
        reason = (
            "Minimum GPU cost satisfying SLO and Actor supply"
            if _meets(best, self.target)
            else "Retain current layout"
            if trigger == "slack"
            else "Recover queue within SLO tolerance"
            if recovering
            and not any(_meets(p, self.target) for p in predictions if _usable(p))
            else "Best request fraction, then Actor deficit and GPU cost"
        )
        return Decision(current, best.layout, tuple(predictions), reason)

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
