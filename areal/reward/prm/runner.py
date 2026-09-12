# SPDX-License-Identifier: Apache-2.0

"""Process-reward runner with all-or-nothing commits per scorer."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal
from urllib.parse import quote

import torch

from areal.api.cli_args import PRMConfig, PRMScorerConfig
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.stats_tracker import ReduceType

logger = logging.getLogger("PRMRunner")


class BaseScorer:
    """Base class for process-reward scorers.

    ``evaluate`` returns a scalar step reward, a dense ``[output_len]`` tensor,
    or ``None`` when the interaction cannot be scored. Scalars are broadcast to
    all output tokens in the turn. The configured weight affects the training
    contribution and reward metrics. Scorers that need additional monitoring
    return typed observations from ``prepare_result()``.

    ``interaction`` and ``ctx`` passed to scorers are read-only inputs.
    """

    name: ClassVar[str] = ""

    def __init__(self, weight: float = 1.0, enabled: bool = True):
        if not self.name:
            raise ValueError(
                f"{type(self).__name__} must define a non-empty class-level name"
            )
        self.weight = float(weight)
        self.enabled = bool(enabled)

    async def score(
        self,
        interaction: InteractionWithTokenLogpReward,
        ctx: Mapping[str, Any],
    ) -> float | torch.Tensor | None:
        """Evaluate one interaction and apply the configured reward weight."""
        result = await self.evaluate(interaction, ctx)
        if result is None:
            return None
        if isinstance(result, torch.Tensor):
            return result * self.weight
        return float(result) * self.weight

    async def evaluate(
        self,
        interaction: InteractionWithTokenLogpReward,
        ctx: Mapping[str, Any],
    ) -> float | torch.Tensor | None:
        """Return a scalar step reward or dense per-output-token rewards."""
        raise NotImplementedError

    def prepare_result(
        self,
        interaction: InteractionWithTokenLogpReward,
        result: float | torch.Tensor,
        ctx: Mapping[str, Any],
    ) -> PRMScorerResult:
        """Attach typed monitoring observations to one unweighted reward result.

        Adapters may override this hook to map a structured scorer result without
        storing request state on a shared scorer instance.
        """
        del interaction, ctx
        return PRMScorerResult(reward=result)

    async def evaluate_result(
        self,
        interaction: InteractionWithTokenLogpReward,
        ctx: Mapping[str, Any],
    ) -> PRMScorerResult | None:
        """Evaluate and prepare one turn in the same async task."""
        result = await self.evaluate(interaction, ctx)
        if result is None:
            return None
        return self.prepare_result(interaction, result, ctx)


class BaseTrajectoryScorer(BaseScorer):
    """Base class for scorers that need the complete ordered trajectory.

    ``evaluate_trajectory`` is called once per runner invocation with interactions
    ordered from ancestors to descendants. It returns raw, unweighted results
    keyed by interaction ID. Omitted IDs receive a zero reward.
    """

    async def evaluate(
        self,
        interaction: InteractionWithTokenLogpReward,
        ctx: Mapping[str, Any],
    ) -> float | torch.Tensor | None:
        del interaction, ctx
        raise RuntimeError(
            f"{type(self).__name__} must be evaluated through PRMRunner as a "
            "trajectory scorer"
        )

    async def evaluate_trajectory(
        self,
        interactions: Sequence[InteractionWithTokenLogpReward],
        ctx: Mapping[str, Any],
    ) -> Mapping[str, float | torch.Tensor | None] | None:
        """Return raw results keyed by interaction ID for one trajectory."""
        raise NotImplementedError


@dataclass(frozen=True)
class PRMMetricObservation:
    """One evaluated, schema-carrying scorer monitoring observation."""

    metric_id: str
    scope: Literal["turn", "trajectory"]
    target_id: str
    value: float | bool
    value_type: Literal["float", "boolean"]
    aggregations: tuple[Literal["count", "rate", "sum", "mean"], ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("metric_id", self.metric_id),
            ("target_id", self.target_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.scope not in {"turn", "trajectory"}:
            raise ValueError(f"unsupported metric scope: {self.scope!r}")
        if self.value_type not in {"float", "boolean"}:
            raise ValueError(f"unsupported metric value type: {self.value_type!r}")
        object.__setattr__(self, "aggregations", tuple(self.aggregations))
        if not self.aggregations or len(set(self.aggregations)) != len(
            self.aggregations
        ):
            raise ValueError("metric aggregations must be non-empty and unique")
        allowed = {"count", "rate"} if self.value_type == "boolean" else {"sum", "mean"}
        unknown = set(self.aggregations).difference(allowed)
        if unknown:
            raise ValueError(
                f"{self.value_type} metric has incompatible aggregations: "
                f"{sorted(unknown)!r}"
            )
        if self.value_type == "boolean":
            if not isinstance(self.value, bool):
                raise TypeError("boolean metric observation value must be bool")
        elif isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise TypeError("float metric observation value must be numeric")
        elif not math.isfinite(float(self.value)):
            raise ValueError("float metric observation value must be finite")


@dataclass(frozen=True)
class PRMScorerResult:
    """One raw reward plus its unweighted monitoring observations."""

    reward: float | torch.Tensor
    observations: tuple[PRMMetricObservation, ...] = ()


@dataclass(frozen=True)
class PRMTurnResult:
    """Committed reward and observations from one scorer interaction."""

    scorer_name: str
    reward: float
    observations: tuple[PRMMetricObservation, ...] = ()


def _metric_key_segment(value: str) -> str:
    """Escape user-defined IDs without creating unbounded metric path levels."""

    return quote(value, safe="-_.:")


def _record_structured_metric(
    tracker: stats_tracker.DistributedStatsTracker,
    scorer_name: str,
    observation: PRMMetricObservation,
) -> None:
    base = "/".join(
        (
            "prm_metric",
            observation.scope,
            _metric_key_segment(scorer_name),
            _metric_key_segment(observation.metric_id),
        )
    )
    denominator = f"{base}/observed_count"
    tracker.denominator(**{denominator: torch.ones(1, dtype=torch.bool)})
    value = torch.tensor([float(observation.value)], dtype=torch.float32)
    for aggregation in observation.aggregations:
        key = f"{base}/{aggregation}"
        if aggregation in {"rate", "mean"}:
            # Rollout workers export independently. SCALAR preserves a local
            # sample count so RolloutController can compute a weighted global
            # average instead of summing per-worker averages.
            tracker.scalar(**{key: float(observation.value)})
        else:
            tracker.stat(denominator, ReduceType.SUM, **{key: value})


def record_prm_results(results: list[PRMTurnResult], *, is_eval: bool = False) -> None:
    """Publish committed turn results without weakening caller-level atomicity."""
    tracker = stats_tracker.get("eval-rollout" if is_eval else "rollout")
    for record in results:
        tracker.scalar(**{f"prm_turn_reward/{record.scorer_name}": record.reward})
        for observation in record.observations:
            _record_structured_metric(tracker, record.scorer_name, observation)


def _resolve_scorer(spec: Any) -> BaseScorer:
    if isinstance(spec, BaseScorer):
        return spec
    if isinstance(spec, Mapping):
        spec = PRMScorerConfig(**spec)
    if isinstance(spec, PRMScorerConfig):
        if not spec.path:
            raise ValueError(
                "PRM scorer path must be a non-empty dotted Python import path"
            )
        cls = import_from_string(spec.path)
        if not isinstance(cls, type) or not issubclass(cls, BaseScorer):
            raise TypeError(
                f"{spec.path!r} resolved to {cls!r}, not a BaseScorer subclass"
            )
        return cls(weight=spec.weight, enabled=spec.enabled, **dict(spec.kwargs))
    raise TypeError(
        f"Unsupported PRM scorer spec type: {type(spec).__name__}; expected "
        "BaseScorer, PRMScorerConfig, or a compatible mapping"
    )


class _ScorerError:
    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


def _order_trajectory_items(
    items: list[tuple[str, InteractionWithTokenLogpReward]],
) -> list[tuple[str, InteractionWithTokenLogpReward]]:
    """Stable-topologically order interactions by their in-cache parent links."""
    remaining = list(items)
    ordered: list[tuple[str, InteractionWithTokenLogpReward]] = []
    while remaining:
        remaining_object_ids = {id(interaction) for _, interaction in remaining}
        ready = [
            item
            for item in remaining
            if item[1].parent is None or id(item[1].parent) not in remaining_object_ids
        ]
        if not ready:
            raise ValueError("Interaction parent links contain a cycle")
        ready_object_ids = {id(interaction) for _, interaction in ready}
        ordered.extend(ready)
        remaining = [item for item in remaining if id(item[1]) not in ready_object_ids]
    return ordered


class PRMRunner:
    """Run configured scorers and commit token rewards atomically per scorer."""

    def __init__(self, config: PRMConfig):
        self.config = config
        self._scorers = (
            [_resolve_scorer(spec) for spec in config.scorers] if config.enabled else []
        )
        self._observation_schemas: dict[
            str,
            dict[
                tuple[str, str],
                tuple[
                    Literal["float", "boolean"],
                    tuple[Literal["count", "rate", "sum", "mean"], ...],
                ],
            ],
        ] = {}
        if not self._scorers:
            logger.warning("PRMRunner has no active scorer configuration")

    @property
    def scorers(self) -> tuple[BaseScorer, ...]:
        return tuple(self._scorers)

    async def run(
        self,
        cache: InteractionCache,
        ctx: Mapping[str, Any] | None = None,
        *,
        is_eval: bool = False,
        record_metrics: bool = True,
    ) -> list[PRMTurnResult]:
        """Score every interaction and return its committed scorer results.

        Metric writes are delayed until every scorer succeeds. Callers that
        compose multiple branch-local runs can set ``record_metrics=False`` and
        publish the returned values only after the whole trajectory succeeds.
        """
        if not self._scorers or not cache:
            return []

        items = list(cache.items())
        for interaction_id, interaction in items:
            if interaction.model_response is None:
                raise ValueError(
                    "PRM requires token-backed interactions; interaction "
                    f"{interaction_id!r} has no model_response"
                )

        score_ctx = dict(ctx or {})
        score_ctx["is_eval"] = is_eval
        committed_results: list[PRMTurnResult] = []
        for scorer in self._scorers:
            if not scorer.enabled:
                continue
            if isinstance(scorer, BaseTrajectoryScorer):
                raw_results = await self._score_trajectory(scorer, items, score_ctx)
                results = [
                    None
                    if result is None
                    else scorer.prepare_result(interaction, result, score_ctx)
                    for (_, interaction), result in zip(items, raw_results, strict=True)
                ]
            else:
                results = await asyncio.gather(
                    *[
                        self._score_one(scorer, interaction, score_ctx)
                        for _, interaction in items
                    ]
                )
            failures = [
                result
                for result in results
                if result is None or isinstance(result, _ScorerError)
            ]
            if failures:
                first_error = next(
                    (
                        result.exc
                        for result in failures
                        if isinstance(result, _ScorerError)
                    ),
                    None,
                )
                if first_error is not None:
                    raise first_error
                raise RuntimeError(
                    f"Scorer {scorer.name!r} returned None for "
                    f"{len(failures)}/{len(results)} interactions"
                )

            prepared_results: list[
                tuple[
                    str,
                    InteractionWithTokenLogpReward,
                    torch.Tensor,
                    PRMTurnResult,
                ]
            ] = []
            observation_keys: set[tuple[str, str, str]] = set()
            observation_schemas: dict[
                tuple[str, str],
                tuple[
                    Literal["float", "boolean"],
                    tuple[Literal["count", "rate", "sum", "mean"], ...],
                ],
            ] = {}
            expected_schemas = self._observation_schemas.get(scorer.name, {})
            for (interaction_id, interaction), result in zip(
                items, results, strict=True
            ):
                assert not isinstance(result, _ScorerError) and result is not None
                if not isinstance(result, PRMScorerResult):
                    raise TypeError(
                        f"{type(scorer).__name__}.evaluate_result() must return "
                        f"PRMScorerResult or None, got {type(result).__name__}"
                    )
                assert interaction.model_response is not None
                output_len = interaction.model_response.output_len
                if output_len <= 0:
                    raise ValueError(
                        f"Interaction {interaction_id!r} has no output token for "
                        f"scorer {scorer.name!r}"
                    )
                raw_result = result.reward
                if isinstance(raw_result, torch.Tensor):
                    raw_result = raw_result.detach().to(
                        device="cpu", dtype=torch.float32
                    )
                    raw_token_rewards = raw_result
                else:
                    raw_result = float(raw_result)
                    raw_token_rewards = torch.full(
                        (output_len,), raw_result, dtype=torch.float32
                    )
                if raw_token_rewards.shape != torch.Size((output_len,)):
                    raise ValueError(
                        f"Scorer {scorer.name!r} must return shape ({output_len},) "
                        f"for interaction {interaction_id!r}, got "
                        f"{tuple(raw_token_rewards.shape)}"
                    )
                if not torch.isfinite(raw_token_rewards).all().item():
                    raise ValueError(
                        f"Scorer {scorer.name!r} produced non-finite token rewards "
                        f"for interaction {interaction_id!r}"
                    )

                contribution = raw_token_rewards * scorer.weight
                if not torch.isfinite(contribution).all().item():
                    raise ValueError(
                        f"Scorer {scorer.name!r} produced non-finite weighted token "
                        f"rewards for interaction {interaction_id!r}"
                    )
                reward_metric = float(contribution.mean().item())
                if not isinstance(result.observations, Sequence) or isinstance(
                    result.observations, (str, bytes)
                ):
                    raise TypeError(
                        f"{type(scorer).__name__} observations must be a sequence"
                    )
                normalized_observations: list[PRMMetricObservation] = []
                for observation in result.observations:
                    if not isinstance(observation, PRMMetricObservation):
                        raise TypeError(
                            f"{type(scorer).__name__} observations must contain "
                            "PRMMetricObservation values"
                        )
                    if (
                        observation.scope == "turn"
                        and observation.target_id != interaction_id
                    ):
                        raise ValueError(
                            f"Turn metric {observation.metric_id!r} targets "
                            f"{observation.target_id!r}, expected {interaction_id!r}"
                        )
                    observation_key = (
                        observation.scope,
                        observation.target_id,
                        observation.metric_id,
                    )
                    if observation_key in observation_keys:
                        raise ValueError(
                            "duplicate structured metric observation for scorer "
                            f"{scorer.name!r}: {observation_key!r}"
                        )
                    observation_keys.add(observation_key)
                    schema_key = (observation.scope, observation.metric_id)
                    schema = (observation.value_type, observation.aggregations)
                    previous_schema = observation_schemas.setdefault(schema_key, schema)
                    if previous_schema != schema:
                        raise ValueError(
                            "inconsistent structured metric schema for scorer "
                            f"{scorer.name!r} metric {schema_key!r}"
                        )
                    normalized_observations.append(observation)

                for schema_key, schema in observation_schemas.items():
                    expected_schema = expected_schemas.get(schema_key)
                    if expected_schema is not None and expected_schema != schema:
                        raise ValueError(
                            "structured metric schema changed across runs for scorer "
                            f"{scorer.name!r} metric {schema_key!r}"
                        )

                prepared_results.append(
                    (
                        interaction_id,
                        interaction,
                        contribution,
                        PRMTurnResult(
                            scorer_name=scorer.name,
                            reward=reward_metric,
                            observations=tuple(normalized_observations),
                        ),
                    )
                )

            updates: list[tuple[str, torch.Tensor, PRMTurnResult]] = []
            for interaction_id, interaction, contribution, result in prepared_results:
                existing = interaction.token_rewards
                if existing is None:
                    combined = contribution.clone()
                else:
                    combined = existing.detach().to(device="cpu", dtype=torch.float32)
                    if combined.shape != contribution.shape:
                        raise ValueError(
                            "Existing token_rewards shape mismatch for interaction "
                            f"{interaction_id!r}: expected "
                            f"{tuple(contribution.shape)}, got {tuple(combined.shape)}"
                        )
                    if not torch.isfinite(combined).all().item():
                        raise ValueError(
                            "Existing token_rewards contain non-finite values for "
                            f"interaction {interaction_id!r}"
                        )
                    combined = combined + contribution
                if not torch.isfinite(combined).all().item():
                    raise ValueError(
                        "Combined token rewards are non-finite for interaction "
                        f"{interaction_id!r} after scorer {scorer.name!r}"
                    )
                updates.append((interaction_id, combined, result))

            # Bind newly observed schemas only after every result and existing
            # reward has passed validation. The publication name is the metric
            # namespace, so same-named scorer instances must share one schema.
            registered_schemas = self._observation_schemas.setdefault(scorer.name, {})
            for schema_key, schema in observation_schemas.items():
                registered_schemas.setdefault(schema_key, schema)

            for interaction_id, token_rewards, result in updates:
                cache.set_token_rewards(interaction_id, token_rewards)
                committed_results.append(result)

        if record_metrics:
            record_prm_results(committed_results, is_eval=is_eval)
        return committed_results

    async def _score_trajectory(
        self,
        scorer: BaseTrajectoryScorer,
        items: list[tuple[str, InteractionWithTokenLogpReward]],
        ctx: Mapping[str, Any],
    ) -> list[float | torch.Tensor | None]:
        ordered_interactions = [
            interaction for _, interaction in _order_trajectory_items(items)
        ]
        results_by_id = await scorer.evaluate_trajectory(ordered_interactions, ctx)
        if results_by_id is None:
            return [None] * len(items)
        if not isinstance(results_by_id, Mapping):
            raise TypeError(
                f"{type(scorer).__name__}.evaluate_trajectory() must return a "
                f"mapping, got {type(results_by_id).__name__}"
            )

        interaction_ids = [interaction.interaction_id for _, interaction in items]
        if any(interaction_id is None for interaction_id in interaction_ids):
            raise ValueError(
                f"Trajectory scorer {scorer.name!r} requires every interaction to "
                "have an ID"
            )
        normalized_ids = [
            interaction_id
            for interaction_id in interaction_ids
            if interaction_id is not None
        ]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(
                f"Trajectory scorer {scorer.name!r} requires unique interaction IDs"
            )
        unknown_ids = set(results_by_id).difference(normalized_ids)
        if unknown_ids:
            raise ValueError(
                f"Trajectory scorer {scorer.name!r} returned unknown interaction "
                f"IDs: {sorted(unknown_ids)!r}"
            )
        return [
            results_by_id.get(interaction_id, 0.0) for interaction_id in normalized_ids
        ]

    async def _score_one(
        self,
        scorer: BaseScorer,
        interaction: InteractionWithTokenLogpReward,
        ctx: Mapping[str, Any],
    ) -> PRMScorerResult | _ScorerError | None:
        try:
            return await scorer.evaluate_result(interaction, ctx)
        except Exception as exc:
            return _ScorerError(exc)
