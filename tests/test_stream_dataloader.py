# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from areal.api.cli_args import TrainDatasetConfig, ValidDatasetConfig
from areal.dataset.stream import StreamDataset
from areal.utils.dataloader import create_dataloader
from areal.utils.stream_dataloader import StreamDataLoader


def make_loader(*, schedule="abababab", batch_size=4, shuffle=False):
    counts = Counter()
    rows = []
    for stream in schedule:
        rows.append({"stream": stream, "id": f"{stream}{counts[stream]}"})
        counts[stream] += 1
    return create_dataloader(
        StreamDataset(rows, stream_key="stream"),
        rank=0,
        world_size=1,
        dataset_config=TrainDatasetConfig(
            batch_size=batch_size, shuffle=shuffle, drop_last=False
        ),
    )


def test_stream_rejections_replenish_three_a_and_one_b_without_advancing_base():
    loader = make_loader()
    initial = [loader.next_rollout(i) for i in range(8)]
    for i, row in enumerate(initial):
        loader.complete_rollout(i, accepted=row["id"] not in {"a0", "a1", "a2", "b0"})

    replacements = [loader.next_rollout(i) for i in range(8, 12)]
    assert [row["stream"] for row in replacements] == ["a", "a", "a", "b"]
    # Reserved replacements are not issued twice while waiting for completion.
    assert loader.next_rollout(12)["stream"] == "a"
    assert loader.next_rollout(13)["stream"] == "b"
    for i in range(8, 12):
        loader.complete_rollout(i, accepted=True)
    assert loader.state_dict()["debt"] == {"a": 0, "b": 0}


def test_replacement_advances_source_and_does_not_steal_next_base_slot():
    loader = make_loader()
    assert loader.next_rollout(0)["id"] == "a0"
    loader.complete_rollout(0, accepted=False)
    assert loader.next_rollout(1)["id"] == "a1"
    assert loader.next_rollout(2)["id"] == "b0"
    loader.complete_rollout(1, accepted=False)
    assert loader.next_rollout(3)["id"] == "a2"
    loader.complete_rollout(3, accepted=True)
    assert loader.next_rollout(4)["id"] == "a3"


def test_repeated_replacement_rejections_keep_one_debt_across_batches():
    loader = make_loader(batch_size=1)
    loader.next_rollout(0)
    loader.complete_rollout(0, accepted=False)
    for task_id in range(1, 12, 2):
        assert loader.next_rollout(task_id)["stream"] == "a"
        loader.complete_rollout(task_id, accepted=False)
        loader.complete_rollout(task_id, accepted=False)
        assert loader.state_dict()["debt"] == {"a": 1, "b": 0}
        # Repeated failures cannot starve the original mixture indefinitely.
        assert loader.next_rollout(task_id + 1)["stream"] == (
            "b" if task_id % 4 == 1 else "a"
        )
        loader.complete_rollout(task_id + 1, accepted=True)
    assert loader.next_rollout(13)["stream"] == "a"
    loader.complete_rollout(13, accepted=True)
    assert loader.next_rollout(14)["stream"] == "b"


def test_replacement_burst_limit_survives_checkpoint_recovery():
    loader = make_loader(batch_size=2)
    for task_id in range(3):
        assert loader.next_rollout(task_id)["stream"] == "a"
        loader.complete_rollout(task_id, accepted=False)
    restored = make_loader(batch_size=2)
    restored.load_state_dict(loader.state_dict())
    for source in (loader, restored):
        assert source.next_rollout(3)["stream"] == "b"
        assert source.state_dict()["debt"] == {"a": 1, "b": 0}
        assert source.next_rollout(4)["stream"] == "a"


def test_large_replacement_backlog_leaves_base_draws_available():
    loader = make_loader(batch_size=2)
    for task_id in range(10):
        loader.next_rollout(task_id)
    for task_id in range(0, 10, 2):
        loader.complete_rollout(task_id, accepted=False)
    # Reservations still leave debt, but the burst limit admits normal B work.
    assert [loader.next_rollout(i)["stream"] for i in range(10, 16)] == [
        "a",
        "a",
        "a",
        "a",
        "a",
        "b",
    ]


@pytest.mark.parametrize("replacement", [False, True])
def test_cancellation_releases_reservation_without_new_debt(replacement):
    loader = make_loader()
    loader.next_rollout(0)
    if replacement:
        loader.complete_rollout(0, accepted=False)
        loader.next_rollout(1)
    loader.complete_rollout(int(replacement), accepted=None)
    assert loader.state_dict()["debt"] == {"a": int(replacement), "b": 0}
    assert loader.next_rollout(2)["stream"] == ("a" if replacement else "b")


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("warmup", [0, 19])
def test_recovery_preserves_buffer_cursors_shuffle_and_unpaid_replacements(
    shuffle, warmup
):
    loader = make_loader(shuffle=shuffle)
    for task_id in range(100, 100 + warmup):
        loader.next_rollout(task_id)
        loader.complete_rollout(task_id, accepted=True)
    rejected = loader.next_rollout(0)["stream"]
    loader.complete_rollout(0, accepted=False)
    assert loader.next_rollout(1)["stream"] == rejected
    saved = loader.state_dict()
    # A cancelled replacement and a process restart must both make debt available.
    loader.complete_rollout(1, accepted=None)
    restored = make_loader(shuffle=shuffle)
    restored.load_state_dict(saved)
    restored.sampler.set_epoch(loader.sampler.epoch)
    for task_id in range(2, 35):
        assert restored.next_rollout(task_id) == loader.next_rollout(task_id)
        loader.complete_rollout(task_id, accepted=True)
        restored.complete_rollout(task_id, accepted=True)
    assert restored.state_dict()["debt"] == {"a": 0, "b": 0}


def test_weighted_base_schedule_is_preserved_after_compensation():
    loader = make_loader(schedule="aaabaaab", batch_size=2)
    loader.next_rollout(0)
    loader.complete_rollout(0, accepted=False)
    assert loader.next_rollout(1)["stream"] == "a"
    loader.complete_rollout(1, accepted=True)
    assert "".join(loader.next_rollout(i)["stream"] for i in range(2, 9)) == "aabaaab"


def test_concurrent_completion_records_each_rejection_once():
    loader = make_loader(schedule="ab" * 40, batch_size=80)
    rows = [loader.next_rollout(i) for i in range(80)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: loader.complete_rollout(i, False), list(range(80)) * 2))
    assert loader.state_dict()["debt"] == dict(Counter(row["stream"] for row in rows))
    with ThreadPoolExecutor(max_workers=8) as pool:
        replacements = list(pool.map(loader.next_rollout, range(80, 160)))
    assert Counter(row["stream"] for row in replacements) == Counter(a=40, b=40)


@pytest.mark.parametrize("rows", [[], [{"stream": ""}], [{"stream": None}], [{}]])
def test_stream_dataset_rejects_invalid_sources(rows):
    with pytest.raises(ValueError):
        StreamDataset(rows, stream_key="stream")


def test_validation_uses_original_rows_without_replacement_sampling():
    rows = [{"stream": "a", "id": 1}, {"stream": "b", "id": 2}]
    loader = create_dataloader(
        StreamDataset(rows, stream_key="stream"),
        rank=0,
        world_size=1,
        dataset_config=ValidDatasetConfig(batch_size=2, shuffle=False),
    )
    assert not isinstance(loader, StreamDataLoader)
    assert list(loader) == [rows]


def test_stream_loader_rejects_unsupported_distributed_sampling():
    dataset = StreamDataset([{"stream": "a"}, {"stream": "b"}], stream_key="stream")
    with pytest.raises(ValueError, match="single rollout controller"):
        create_dataloader(
            dataset,
            rank=0,
            world_size=2,
            dataset_config=TrainDatasetConfig(batch_size=2),
        )


def test_restore_rejects_incompatible_source_membership():
    loader = make_loader()
    with pytest.raises(ValueError, match="does not match"):
        make_loader(schedule="acacacac").load_state_dict(loader.state_dict())


def test_restore_flat_checkpoint_explains_missing_stream_state():
    loader = make_loader()
    with pytest.raises(ValueError, match="per-stream cursors and replacement debt"):
        loader.load_state_dict(loader.state_dict()["base"])
