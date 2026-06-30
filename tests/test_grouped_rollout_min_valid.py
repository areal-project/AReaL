import asyncio
import logging

import pytest
import torch

from areal.infra.remote_inf_engine import GroupedRolloutWorkflow


class _FakeWorkflow:
    def __init__(self, n_none: int):
        self._n_none = n_none
        self._calls = 0

    async def arun_episode(self, engine, data):
        self._calls += 1
        if self._calls <= self._n_none:
            return None
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


def _run(group_size, n_none, min_valid_group_size):
    wf = GroupedRolloutWorkflow(
        _FakeWorkflow(n_none),
        group_size=group_size,
        logger=logging.getLogger("test"),
        min_valid_group_size=min_valid_group_size,
    )
    return asyncio.run(wf.arun_episode(engine=None, data={}))


@pytest.mark.parametrize(
    "min_valid, n_none, kept",
    [
        (1, 1, True),
        (2, 2, True),
        (2, 3, False),
        (4, 1, False),
    ],
)
def test_min_valid_group_size_threshold(min_valid, n_none, kept):
    out = _run(group_size=4, n_none=n_none, min_valid_group_size=min_valid)
    assert (out is not None) == kept
    if kept:
        assert out["input_ids"].shape[0] == 4 - n_none


def test_all_none_returns_none():
    assert _run(group_size=4, n_none=4, min_valid_group_size=1) is None


@pytest.mark.parametrize("bad", [0, 5])
def test_threshold_outside_valid_range_raises(bad):
    with pytest.raises(ValueError, match="min_valid_group_size must be in"):
        GroupedRolloutWorkflow(
            _FakeWorkflow(0),
            group_size=4,
            logger=logging.getLogger("test"),
            min_valid_group_size=bad,
        )
