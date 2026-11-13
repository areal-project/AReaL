# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples.swe import train_swe_rl
from examples.swe.utils import ArenaStreamConfig, SWEEnvConfig

import areal
from areal.dataset.arena_stream import ArenaStreamDataset


@pytest.mark.parametrize(
    "version,single_controller,stream_count,enabled",
    [
        ("v1", True, 2, True),
        ("v1", True, 1, False),
        ("v2", True, 2, False),
        ("v1", False, 2, False),
    ],
)
def test_arena_entrypoint_enables_compensation_only_for_supported_training(
    monkeypatch, version, single_controller, stream_count, enabled
):
    streams = [
        ArenaStreamConfig(name=str(i), stream_id=str(i)) for i in range(stream_count)
    ]
    rows = [{"arena_stream_name": stream.name} for stream in streams]
    config = SimpleNamespace(
        econfig=SWEEnvConfig(dataset_source="arena"),
        scheduler=SimpleNamespace(type="local"),
        rollout=SimpleNamespace(_version=version),
        train_dataset=SimpleNamespace(batch_size=2),
        gconfig=SimpleNamespace(
            temperature=1.0, top_p=1.0, top_k=-1, max_new_tokens=32
        ),
        experiment_name="test",
        trial_name="stream",
    )
    trainer = MagicMock()
    monkeypatch.setitem(areal.__dict__, "PPOTrainer", trainer)
    monkeypatch.setattr(train_swe_rl, "load_expr_config", lambda *args: (config, None))
    monkeypatch.setattr(train_swe_rl, "is_single_controller", lambda: single_controller)
    monkeypatch.setattr(
        train_swe_rl,
        "get_arena_mixture_dataset",
        lambda *args, **kwargs: (rows, streams),
    )

    train_swe_rl.main([])

    datasets = trainer.call_args.kwargs
    assert isinstance(datasets["train_dataset"], ArenaStreamDataset) == enabled
    assert datasets["valid_dataset"] is rows
    if not enabled:
        assert datasets["train_dataset"] is rows
