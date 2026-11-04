#!/usr/bin/env python3
"""Small GPU-host smoke check for consumed rollout group timing metrics.

Run from the repository root after installing AReaL:
    python3 scripts/check_rollout_filter_timeperf_minimal.py
"""

import asyncio
from unittest.mock import MagicMock

from areal.api import RolloutWorkflow
from areal.api.cli_args import InferenceEngineConfig
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_executor import WorkflowExecutor, _RolloutTaskInput
from areal.utils.rollout_metrics import ConsumedMetrics


class _SyntheticWorkflow(RolloutWorkflow):
    def __init__(self, trajectory):
        self.trajectory = trajectory

    async def arun_episode(self, engine, data):
        return self.trajectory


async def main() -> None:
    executor = WorkflowExecutor(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            consumer_batch_size=1,
            check_trajectory_format=False,
        ),
        inference_engine=MagicMock(),
        staleness_manager=MagicMock(),
    )
    executor.logger = MagicMock()
    consumed = ConsumedMetrics()
    cases = (
        (_SyntheticWorkflow({"payload": 1}), None),
        (_SyntheticWorkflow({"payload": 1}), lambda _: False),
        (
            GroupedRolloutWorkflow(
                _SyntheticWorkflow(None),
                group_size=2,
                min_usable_group_size=2,
                logger=MagicMock(),
            ),
            None,
        ),
        (_SyntheticWorkflow(None), None),
    )
    for task_id, (workflow, filter_fn) in enumerate(cases):
        task = executor._create_workflow_task(
            _RolloutTaskInput(
                task_id=task_id,
                data={},
                workflow=workflow,
                should_accept_fn=filter_fn,
            )
        )
        consumed.consume(await task())

    stats = consumed.export()
    assert stats["timeperf/rollout_group_attempt_seconds"] > 0
    for reason in ("filter_rejected", "below_min", "other_discarded"):
        assert stats[f"timeperf/rollout_group_{reason}_seconds"] > 0
        assert 0 < stats[f"timeperf/rollout_group_{reason}_share"] < 1
    assert (
        sum(
            stats[f"timeperf/rollout_group_{reason}_share"]
            for reason in ("filter_rejected", "below_min", "other_discarded")
        )
        < 1
    )
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
