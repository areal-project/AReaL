# SPDX-License-Identifier: Apache-2.0

"""Dependency-free example scorer for multi-turn agent training."""

from collections.abc import Mapping
from typing import Any

from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.reward.prm import BaseScorer, PRMMetricObservation, PRMScorerResult


class LengthBudgetScorer(BaseScorer):
    """Score each turn as one within the output budget and zero above it.

    These bounded scores can be used with process-weighted advantages. The
    budget is independent of the generation limit, so exceeding it does not
    truncate a response.
    """

    name = "length_budget"

    def __init__(
        self, max_output_tokens: int = 128, weight: float = 1.0, enabled: bool = True
    ):
        super().__init__(weight=weight, enabled=enabled)
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.max_output_tokens = max_output_tokens

    async def evaluate(
        self, interaction: InteractionWithTokenLogpReward, ctx: Mapping[str, Any]
    ) -> float | None:
        if interaction.model_response is None:
            return None
        return float(interaction.model_response.output_len <= self.max_output_tokens)

    def prepare_result(
        self,
        interaction: InteractionWithTokenLogpReward,
        result: float,
        ctx: Mapping[str, Any],
    ) -> PRMScorerResult:
        assert interaction.interaction_id is not None
        return PRMScorerResult(
            reward=result,
            observations=(
                PRMMetricObservation(
                    metric_id="within_budget",
                    scope="turn",
                    target_id=interaction.interaction_id,
                    value=bool(result),
                    value_type="boolean",
                    aggregations=("count", "rate"),
                ),
            ),
        )
