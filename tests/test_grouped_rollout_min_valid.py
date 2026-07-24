import asyncio
import logging

import pytest
import torch

from areal.api import RolloutWorkflow
from areal.api.cli_args import InferenceEngineConfig
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow, RemoteInfEngine


class _Workflow(RolloutWorkflow):
    def __init__(self, none_count: int):
        self._none_count = none_count
        self._calls = 0

    async def arun_episode(self, engine, data):
        self._calls += 1
        if self._calls <= self._none_count:
            return None
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def _run_group(none_count: int, min_valid_group_size: int):
    workflow = GroupedRolloutWorkflow(
        _Workflow(none_count),
        group_size=4,
        logger=logging.getLogger("test"),
        min_valid_group_size=min_valid_group_size,
    )
    return asyncio.run(workflow.arun_episode(engine=None, data={}))


@pytest.mark.parametrize(
    ("min_valid_group_size", "none_count", "expected_rows"),
    [
        (1, 4, None),
        (1, 1, 3),
        (2, 2, 2),
        (2, 3, None),
        (4, 1, None),
    ],
)
def test_min_valid_group_size_filters_underfilled_groups(
    min_valid_group_size, none_count, expected_rows
):
    out = _run_group(none_count, min_valid_group_size)

    if expected_rows is None:
        assert out is None
    else:
        assert out is not None
        assert out["input_ids"].shape[0] == expected_rows


@pytest.mark.parametrize("min_valid_group_size", [0, 5])
def test_min_valid_group_size_outside_group_bounds_raises(min_valid_group_size):
    with pytest.raises(ValueError, match="min_valid_group_size must be in"):
        GroupedRolloutWorkflow(
            _Workflow(0),
            group_size=4,
            logger=logging.getLogger("test"),
            min_valid_group_size=min_valid_group_size,
        )


def test_configured_min_valid_group_size_reaches_grouped_workflow():
    engine = object.__new__(RemoteInfEngine)
    engine.config = InferenceEngineConfig(backend="sglang:d1", min_valid_group_size=3)
    engine.logger = logging.getLogger("test")

    resolved = engine._resolve_workflow(
        _Workflow(none_count=0), workflow_kwargs=None, group_size=4
    )

    assert isinstance(resolved, GroupedRolloutWorkflow)
    assert resolved.min_valid_group_size == 3
