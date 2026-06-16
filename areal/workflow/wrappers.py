# SPDX-License-Identifier: Apache-2.0

"""Workflow wrappers that adapt rollout execution behavior."""

from __future__ import annotations

import asyncio
from logging import Logger
from typing import Any

from areal.api import InferenceEngine, RolloutWorkflow
from areal.utils.data import concat_padded_tensors


class GroupedRolloutWorkflow(RolloutWorkflow):
    """Run a wrapped workflow multiple times for one input and merge results."""

    def __init__(
        self,
        workflow: RolloutWorkflow,
        group_size: int,
        logger: Logger,
    ):
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.workflow = workflow
        self.group_size = group_size
        self.logger = logger

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any] | None:
        from areal.experimental.openai import InteractionWithTokenLogpReward

        results = await asyncio.gather(
            *[self.workflow.arun_episode(engine, data) for _ in range(self.group_size)]
        )

        valid_results = [r for r in results if r is not None]

        # All results None -> return None
        if not valid_results:
            return None

        # Some results None -> warn and continue with valid ones
        if len(valid_results) < len(results):
            self.logger.warning(
                f"GroupedRolloutWorkflow: {len(results) - len(valid_results)}/{len(results)} "
                "trajectories returned None, using remaining results"
            )

        # Check if results are InteractionWithTokenLogpReward dicts
        first = valid_results[0]
        if (
            isinstance(first, dict)
            and first
            and all(
                isinstance(v, InteractionWithTokenLogpReward) for v in first.values()
            )
        ):
            # Merge dicts - each result is {completion_id: InteractionWithTokenLogpReward}
            merged: dict[str, InteractionWithTokenLogpReward] = {}
            for result in valid_results:
                merged.update(result)
            return merged if merged else None

        # Otherwise, tensor dicts - concatenate
        concatenated = concat_padded_tensors(valid_results)
        return concatenated if concatenated else None
