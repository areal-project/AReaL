"""Small CPU-only tests for process-reward scoring and commit semantics."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from examples.prm.scorers import LengthBudgetScorer

from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.reward.prm import (
    BaseScorer,
    BaseTrajectoryScorer,
    PRMConfig,
    PRMMetricObservation,
    PRMRunner,
    PRMScorerResult,
)
from areal.utils import stats_tracker


@pytest.mark.asyncio
@pytest.mark.parametrize(("output_len", "expected"), [(3, 1.0), (4, 0.0)])
async def test_length_budget_scorer_produces_bounded_rewards_and_observations(
    output_len, expected
):
    interaction = _interaction("turn", output_len=output_len)
    cache = InteractionCache.from_dict({"turn": interaction})
    runner = PRMRunner(PRMConfig(scorers=[LengthBudgetScorer(max_output_tokens=3)]))
    results = await runner.run(cache, record_metrics=False)
    torch.testing.assert_close(
        interaction.token_rewards,
        torch.full((output_len,), expected),
        rtol=0.0,
        atol=0.0,
    )
    assert results[0].observations[0].value is bool(expected)


def _interaction(interaction_id: str, output_len: int = 3):
    response = MagicMock()
    response.output_len = output_len
    interaction = InteractionWithTokenLogpReward(
        model_response=response,
        reward=0.25,
        chat_template_type="concat",
        messages=[{"role": "user", "content": "fix it"}],
        output_message_list=[{"role": "assistant", "content": "done"}],
    )
    interaction._interaction_id = interaction_id
    return interaction


class _ScalarScorer(BaseScorer):
    name = "scalar"

    def __init__(self, value: float, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    async def evaluate(self, interaction, ctx):
        return self.value


@pytest.mark.asyncio
async def test_runner_records_weighted_reward():
    stats_tracker.export_all(reduce_group=None)
    interaction = _interaction("turn-1")
    cache = InteractionCache.from_dict({"turn-1": interaction})
    runner = PRMRunner(PRMConfig(scorers=[_ScalarScorer(value=-1.0, weight=0.2)]))

    records = await runner.run(cache)

    assert interaction.reward == pytest.approx(0.25)
    torch.testing.assert_close(
        interaction.token_rewards,
        torch.tensor([-0.2, -0.2, -0.2]),
        rtol=0.0,
        atol=0.0,
    )
    metrics = stats_tracker.export_all(reduce_group=None)
    assert metrics["rollout/prm_turn_reward/scalar"] == pytest.approx(-0.2)
    assert records[0].reward == pytest.approx(-0.2)
    assert records[0].observations == ()


@pytest.mark.asyncio
async def test_uniform_dense_result_logs_mean_weighted_reward():
    class _UniformDenseScorer(BaseScorer):
        name = "uniform_dense"

        async def evaluate(self, interaction, ctx):
            return torch.full((interaction.model_response.output_len,), -0.2)

    stats_tracker.export_all(reduce_group=None)
    runner = PRMRunner(PRMConfig(scorers=[_UniformDenseScorer()]))
    await runner.run(InteractionCache.from_dict({"turn-1": _interaction("turn-1")}))

    metrics = stats_tracker.export_all(reduce_group=None)
    assert metrics["rollout/prm_turn_reward/uniform_dense"] == pytest.approx(-0.2)


@pytest.mark.parametrize(
    "cache_order",
    [
        ("a", "b", "c"),
        ("c", "b", "a"),
        ("b", "a", "c"),
    ],
)
@pytest.mark.asyncio
async def test_trajectory_scorer_orders_turns_and_targets_selected_turn(cache_order):
    class _Trajectory(BaseTrajectoryScorer):
        name = "trajectory"

        async def evaluate_trajectory(self, interactions, ctx):
            assert [interaction.interaction_id for interaction in interactions] == [
                "a",
                "b",
                "c",
            ]
            return {"c": 0.75}

    root = _interaction("a")
    middle = _interaction("b")
    leaf = _interaction("c")
    middle.parent = root
    leaf.parent = middle
    interactions = {"a": root, "b": middle, "c": leaf}
    cache = InteractionCache.from_dict(
        {interaction_id: interactions[interaction_id] for interaction_id in cache_order}
    )

    await PRMRunner(PRMConfig(scorers=[_Trajectory()])).run(cache)

    torch.testing.assert_close(cache["a"].token_rewards, torch.zeros(3))
    torch.testing.assert_close(cache["b"].token_rewards, torch.zeros(3))
    torch.testing.assert_close(cache["c"].token_rewards, torch.full((3,), 0.75))


@pytest.mark.asyncio
async def test_runner_records_boolean_observations_as_counts_and_rates():
    class _ObservedFlagScorer(_ScalarScorer):
        name = "flagged"

        async def evaluate_result(self, interaction, ctx):
            del ctx
            return PRMScorerResult(
                reward=self.value,
                observations=(
                    PRMMetricObservation(
                        metric_id="failed_turn",
                        scope="turn",
                        target_id=interaction.interaction_id,
                        value=True,
                        value_type="boolean",
                        aggregations=("count", "rate"),
                    ),
                ),
            )

    stats_tracker.export_all(reduce_group=None)
    runner = PRMRunner(PRMConfig(scorers=[_ObservedFlagScorer(value=-1.0, weight=0.0)]))

    records = await runner.run(
        InteractionCache.from_dict({"turn-1": _interaction("turn-1")})
    )

    assert records[0].reward == 0.0
    assert records[0].observations[0].metric_id == "failed_turn"
    assert records[0].observations[0].value is True
    metrics = stats_tracker.export_all(reduce_group=None)
    assert metrics["rollout/prm_turn_reward/flagged"] == 0.0
    prefix = "rollout/prm_metric/turn/flagged/failed_turn"
    assert metrics[f"{prefix}/observed_count"] == 1.0
    assert metrics[f"{prefix}/count"] == 1.0
    assert metrics[f"{prefix}/rate"] == 1.0


@pytest.mark.asyncio
async def test_runner_accepts_adapter_prepared_reward_and_observations():
    """A host adapter can carry typed observations with the reward."""

    class _PreparedScorer(_ScalarScorer):
        name = "prepared"

        async def evaluate_result(self, interaction, ctx):
            del interaction, ctx
            return PRMScorerResult(
                reward=-0.5,
                observations=(
                    PRMMetricObservation(
                        metric_id="typed_flag",
                        scope="turn",
                        target_id="turn-1",
                        value=False,
                        value_type="boolean",
                        aggregations=("count", "rate"),
                    ),
                ),
            )

    interaction = _interaction("turn-1")
    runner = PRMRunner(PRMConfig(scorers=[_PreparedScorer(value=99.0, weight=0.2)]))

    records = await runner.run(
        InteractionCache.from_dict({"turn-1": interaction}),
        record_metrics=False,
    )

    torch.testing.assert_close(
        interaction.token_rewards,
        torch.full((3,), -0.1),
        rtol=0.0,
        atol=0.0,
    )
    assert records[0].observations[0].metric_id == "typed_flag"
    assert records[0].observations[0].value is False


@pytest.mark.asyncio
async def test_runner_aggregates_structured_float_and_boolean_observations():
    """Core metrics keep declared cardinality and aggregation semantics."""

    class _ObservedScorer(_ScalarScorer):
        name = "observed"

        async def evaluate_result(self, interaction, ctx):
            del ctx
            index = 1 if interaction.interaction_id == "turn-1" else 3
            return PRMScorerResult(
                reward=0.0,
                observations=(
                    PRMMetricObservation(
                        metric_id="quality/value",
                        scope="turn",
                        target_id=interaction.interaction_id,
                        value=float(index),
                        value_type="float",
                        aggregations=("sum", "mean"),
                    ),
                    PRMMetricObservation(
                        metric_id="accepted",
                        scope="turn",
                        target_id=interaction.interaction_id,
                        value=index == 1,
                        value_type="boolean",
                        aggregations=("count", "rate"),
                    ),
                ),
            )

    stats_tracker.export_all(reduce_group=None)
    runner = PRMRunner(PRMConfig(scorers=[_ObservedScorer(value=0.0)]))

    await runner.run(
        InteractionCache.from_dict(
            {
                "turn-1": _interaction("turn-1"),
                "turn-2": _interaction("turn-2"),
            }
        )
    )

    metrics = stats_tracker.export_all(reduce_group=None)
    prefix = "rollout/prm_metric/turn/observed"
    assert metrics[f"{prefix}/quality%2Fvalue/observed_count"] == 2.0
    assert metrics[f"{prefix}/quality%2Fvalue/sum"] == pytest.approx(4.0)
    assert metrics[f"{prefix}/quality%2Fvalue/mean"] == pytest.approx(2.0)
    assert metrics[f"{prefix}/quality%2Fvalue/mean__count"] == 2
    assert metrics[f"{prefix}/accepted/observed_count"] == 2.0
    assert metrics[f"{prefix}/accepted/count"] == 1.0
    assert metrics[f"{prefix}/accepted/rate"] == pytest.approx(0.5)
    assert metrics[f"{prefix}/accepted/rate__count"] == 2


@pytest.mark.asyncio
async def test_runner_rejects_duplicate_trajectory_observation_before_commit():
    class _DuplicateObservationScorer(_ScalarScorer):
        name = "duplicate_observation"

        async def evaluate_result(self, interaction, ctx):
            del interaction, ctx
            return PRMScorerResult(
                reward=0.0,
                observations=(
                    PRMMetricObservation(
                        metric_id="trajectory_quality",
                        scope="trajectory",
                        target_id="trajectory-1",
                        value=1.0,
                        value_type="float",
                        aggregations=("mean",),
                    ),
                ),
            )

    interactions = {
        "turn-1": _interaction("turn-1"),
        "turn-2": _interaction("turn-2"),
    }
    runner = PRMRunner(PRMConfig(scorers=[_DuplicateObservationScorer(value=0.0)]))

    with pytest.raises(ValueError, match="duplicate structured metric"):
        await runner.run(InteractionCache.from_dict(interactions))

    assert all(item.token_rewards is None for item in interactions.values())


@pytest.mark.asyncio
async def test_runner_rejects_turn_observation_for_another_interaction():
    class _WrongTargetScorer(_ScalarScorer):
        name = "wrong_target"

        async def evaluate_result(self, interaction, ctx):
            del interaction, ctx
            return PRMScorerResult(
                reward=0.0,
                observations=(
                    PRMMetricObservation(
                        metric_id="quality",
                        scope="turn",
                        target_id="another-turn",
                        value=1.0,
                        value_type="float",
                        aggregations=("mean",),
                    ),
                ),
            )

    interaction = _interaction("turn-1")

    with pytest.raises(ValueError, match="expected 'turn-1'"):
        await PRMRunner(PRMConfig(scorers=[_WrongTargetScorer(value=0.0)])).run(
            InteractionCache.from_dict({"turn-1": interaction})
        )

    assert interaction.token_rewards is None


@pytest.mark.asyncio
async def test_failed_run_does_not_bind_structured_metric_schema():
    class _MutableSchemaScorer(_ScalarScorer):
        name = "mutable_schema"

        def __init__(self):
            super().__init__(value=0.0)
            self.aggregation = "mean"

        async def evaluate_result(self, interaction, ctx):
            del ctx
            return PRMScorerResult(
                reward=0.0,
                observations=(
                    PRMMetricObservation(
                        metric_id="quality",
                        scope="turn",
                        target_id=interaction.interaction_id,
                        value=1.0,
                        value_type="float",
                        aggregations=(self.aggregation,),
                    ),
                ),
            )

    scorer = _MutableSchemaScorer()
    runner = PRMRunner(PRMConfig(scorers=[scorer]))
    invalid = _interaction("turn-1")
    invalid.token_rewards = torch.zeros(2)

    with pytest.raises(ValueError, match="Existing token_rewards shape mismatch"):
        await runner.run(InteractionCache.from_dict({"turn-1": invalid}))

    scorer.aggregation = "sum"
    valid = _interaction("turn-1")
    await runner.run(InteractionCache.from_dict({"turn-1": valid}))

    torch.testing.assert_close(
        valid.token_rewards,
        torch.zeros(3),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.asyncio
async def test_same_named_scorers_share_published_metric_schema():
    class _NamedSchemaScorer(_ScalarScorer):
        name = "shared_namespace"

        def __init__(self, aggregation):
            super().__init__(value=0.0)
            self.aggregation = aggregation

        async def evaluate_result(self, interaction, ctx):
            del ctx
            return PRMScorerResult(
                reward=0.0,
                observations=(
                    PRMMetricObservation(
                        metric_id="quality",
                        scope="turn",
                        target_id=interaction.interaction_id,
                        value=1.0,
                        value_type="float",
                        aggregations=(self.aggregation,),
                    ),
                ),
            )

    interaction = _interaction("turn-1")
    runner = PRMRunner(
        PRMConfig(
            scorers=[
                _NamedSchemaScorer("mean"),
                _NamedSchemaScorer("sum"),
            ]
        )
    )

    with pytest.raises(ValueError, match="schema changed across runs"):
        await runner.run(InteractionCache.from_dict({"turn-1": interaction}))

    torch.testing.assert_close(
        interaction.token_rewards,
        torch.zeros(3),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.asyncio
async def test_runner_validates_observation_types_before_committing_scorer():
    class _InvalidObservationScorer(_ScalarScorer):
        name = "invalid_observation"

        async def evaluate_result(self, interaction, ctx):
            del interaction, ctx
            return PRMScorerResult(
                reward=-1.0,
                observations=(object(),),  # type: ignore[arg-type]
            )

    interaction = _interaction("turn-1")
    runner = PRMRunner(PRMConfig(scorers=[_InvalidObservationScorer(value=-1.0)]))

    with pytest.raises(TypeError, match="must contain PRMMetricObservation"):
        await runner.run(InteractionCache.from_dict({"turn-1": interaction}))

    assert interaction.token_rewards is None


@pytest.mark.asyncio
async def test_runner_is_all_or_nothing_within_one_scorer():
    class _PartialScorer(BaseScorer):
        name = "partial"

        async def evaluate(self, interaction, ctx):
            if interaction.interaction_id == "turn-2":
                return None
            return -1.0

    interactions = {
        "turn-1": _interaction("turn-1"),
        "turn-2": _interaction("turn-2"),
    }
    runner = PRMRunner(PRMConfig(scorers=[_PartialScorer()]))

    with pytest.raises(RuntimeError, match="returned None"):
        await runner.run(InteractionCache.from_dict(interactions))

    assert all(item.token_rewards is None for item in interactions.values())


@pytest.mark.asyncio
async def test_runner_validates_all_shapes_before_committing_scorer():
    """One malformed result must not leave earlier interactions modified."""

    class _ShapeScorer(BaseScorer):
        name = "shape"

        async def evaluate(self, interaction, ctx):
            del ctx
            size = 2 if interaction.interaction_id == "turn-2" else 3
            return torch.zeros(size)

    interactions = {
        "turn-1": _interaction("turn-1"),
        "turn-2": _interaction("turn-2"),
    }
    runner = PRMRunner(PRMConfig(scorers=[_ShapeScorer()]))

    with pytest.raises(ValueError, match=r"must return shape \(3,\)"):
        await runner.run(InteractionCache.from_dict(interactions))

    assert all(item.token_rewards is None for item in interactions.values())


@pytest.mark.asyncio
async def test_runner_validates_existing_rewards_before_any_commit():
    """Malformed prior state must not partially commit the next scorer."""
    interactions = {
        "turn-1": _interaction("turn-1"),
        "turn-2": _interaction("turn-2"),
    }
    malformed = torch.tensor([0.1, 0.2])
    interactions["turn-2"].token_rewards = malformed
    runner = PRMRunner(PRMConfig(scorers=[_ScalarScorer(value=0.5)]))

    with pytest.raises(ValueError, match="Existing token_rewards shape mismatch"):
        await runner.run(InteractionCache.from_dict(interactions))

    assert interactions["turn-1"].token_rewards is None
    assert interactions["turn-2"].token_rewards is malformed


@pytest.mark.asyncio
async def test_runner_rejects_non_finite_dense_rewards():
    class _DenseScorer(BaseScorer):
        name = "dense"

        async def evaluate(self, interaction, ctx):
            return torch.tensor([0.0, float("nan"), 0.0])

    interaction = _interaction("turn-1")
    runner = PRMRunner(PRMConfig(scorers=[_DenseScorer()]))
    with pytest.raises(ValueError, match="non-finite"):
        await runner.run(InteractionCache.from_dict({"turn-1": interaction}))
    assert interaction.token_rewards is None


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
async def test_runner_rejects_non_finite_scalar_rewards(value):
    """NaN and infinity must be rejected before entering training tensors."""
    interaction = _interaction("turn-1")
    runner = PRMRunner(PRMConfig(scorers=[_ScalarScorer(value=value)]))

    with pytest.raises(ValueError, match="non-finite"):
        await runner.run(InteractionCache.from_dict({"turn-1": interaction}))

    assert interaction.token_rewards is None


@pytest.mark.asyncio
async def test_runner_rejects_message_only_interaction():
    interaction = InteractionWithTokenLogpReward(
        model_response=None,
        output_message_list=[{"role": "assistant", "content": "external response"}],
    )
    interaction._interaction_id = "message-only"
    runner = PRMRunner(PRMConfig(scorers=[_ScalarScorer(value=0.0)]))

    with pytest.raises(ValueError, match="token-backed interactions"):
        await runner.run(InteractionCache.from_dict({"message-only": interaction}))


@pytest.mark.asyncio
async def test_runner_routes_eval_metric_to_eval_scope():
    stats_tracker.export_all(reduce_group=None)
    runner = PRMRunner(PRMConfig(scorers=[_ScalarScorer(value=0.5)]))

    await runner.run(
        InteractionCache.from_dict({"turn-1": _interaction("turn-1")}),
        is_eval=True,
    )

    metrics = stats_tracker.export_all(reduce_group=None)
    assert metrics["eval-rollout/prm_turn_reward/scalar"] == pytest.approx(0.5)
    assert "rollout/prm_turn_reward/scalar" not in metrics
