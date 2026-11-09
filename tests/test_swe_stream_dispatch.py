# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from examples.swe import train_swe_rl
from examples.swe.utils import ArenaStreamConfig, SWEEnvConfig

import areal
from areal.dataset.stream import StreamDataset


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
    assert isinstance(datasets["train_dataset"], StreamDataset) == enabled
    assert datasets["valid_dataset"] is rows
    if not enabled:
        assert datasets["train_dataset"] is rows


@pytest.mark.parametrize(
    "version,single_controller,stream_count,enabled",
    [
        ("v1", True, 3, True),
        ("v1", True, 1, False),
        ("v2", True, 3, False),
        ("v1", False, 3, False),
    ],
)
def test_qwen_recipe_uses_stream_feedback_with_filter_disabled(
    monkeypatch, version, single_controller, stream_count, enabled
):
    from datasets import Dataset

    from examples.swe.qwen38_flash_next import train_rl
    from examples.swe.utils import SWEPPOConfig

    from areal.api import cli_args
    from areal.utils.dataloader import create_dataloader
    from areal.utils.stream_dataloader import StreamDataLoader

    streams = [
        ArenaStreamConfig(name=str(i), stream_id=str(i)) for i in range(stream_count)
    ]
    rows = Dataset.from_list(
        [
            {"arena_stream_name": str(i % stream_count), "data_id": str(i)}
            for i in range(64)
        ]
    )
    config = SWEPPOConfig()
    config.rollout._version = version
    config.train_dataset.batch_size = 64
    config.train_dataset.num_workers = 0
    config.should_accept_fn = None
    monkeypatch.setattr(cli_args, "load_expr_config", lambda *_: (config, None))
    monkeypatch.setattr(train_swe_rl, "is_single_controller", lambda: single_controller)
    monkeypatch.setattr(
        train_swe_rl, "get_arena_mixture_dataset", lambda *_, **__: (rows, streams)
    )
    monkeypatch.delenv("QWEN_ARENA_TASK_IDS_FILE", raising=False)
    monkeypatch.delenv("QWEN_REQUIRE_RECOVER", raising=False)
    monkeypatch.delenv("QWEN_EQUAL_STREAM_SAMPLING", raising=False)
    captured = []

    class CaptureTrainer:
        def __init__(self, config, train_dataset, valid_dataset):
            assert valid_dataset is None
            self.dataset = train_dataset
            self.recover_info = None
            captured.append(train_dataset)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def train(self, **kwargs):
            assert kwargs["dynamic_filter_fn"] is None
            if enabled:
                loader = create_dataloader(self.dataset, 0, 1, config.train_dataset)
                assert isinstance(loader, StreamDataLoader)
                failed = loader.next_rollout(1)
                loader.complete_rollout(1, False)
                replacement = loader.next_rollout(2)
                assert replacement["arena_stream_name"] == failed["arena_stream_name"]
                loader.complete_rollout(2, True)
                assert not any(loader.state_dict()["debt"].values())

    monkeypatch.setattr(areal, "PPOTrainer", CaptureTrainer)
    train_rl.main("swe", [])
    assert isinstance(captured[0], StreamDataset) == enabled
    if not enabled:
        assert captured[0] is rows


def test_equal_qwen_stream_sampling_preserves_all_rows_and_source_replacement(
    monkeypatch,
):
    from collections import Counter

    from examples.swe.qwen38_flash_next.train_rl import prepare_arena_training_dataset
    from examples.swe.utils import SWEPPOConfig

    from areal.utils.dataloader import create_dataloader

    rows = [
        {"arena_stream_name": stream, "data_id": f"{stream}-{i}"}
        for stream, size in (("a", 7), ("b", 3), ("c", 2))
        for i in range(size)
    ]
    config = SWEPPOConfig()
    config.rollout._version = "v1"
    config.train_dataset.batch_size = 4
    config.train_dataset.num_workers = 0
    config.train_dataset.shuffle = False
    monkeypatch.setenv("QWEN_EQUAL_STREAM_SAMPLING", "1")
    monkeypatch.setattr(train_swe_rl, "is_single_controller", lambda: True)
    dataset = prepare_arena_training_dataset(rows, ["a", "b", "c"], config)
    assert dataset.dataset is rows
    assert Counter(dataset.streams) == {"a": 8, "b": 8, "c": 8}
    loader = create_dataloader(dataset, 0, 1, config.train_dataset)
    assert len(loader) == 6
    seen = set()
    counts = Counter()
    for task_id in range(24):
        row = loader.next_rollout(task_id)
        seen.add(row["data_id"])
        counts[row["arena_stream_name"]] += 1
        loader.complete_rollout(task_id, True)
    assert counts == {"a": 8, "b": 8, "c": 8}
    assert seen == {row["data_id"] for row in rows}
    failed = loader.next_rollout(24)
    loader.complete_rollout(24, False)
    assert loader.next_rollout(25)["arena_stream_name"] == failed["arena_stream_name"]
    loader.complete_rollout(25, True)
    assert not any(loader.state_dict()["debt"].values())
