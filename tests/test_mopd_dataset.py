# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.api.cli_args import DatasetSourceConfig, TrainDatasetConfig
from areal.dataset.mopd import (
    MOPD_ROUTE_METADATA_KEY,
    MOPDDataset,
    get_mopd_dataset,
    is_remote_dataset,
)
from areal.infra.data_service.rdataset import RDataset


def test_mopd_dataset_routes_each_source_without_mutating_samples():
    """Every item inherits only its source route and stored samples stay unchanged."""
    samples = {
        "math": [{"messages": ["m0"]}, {"messages": ["m1"]}],
        "code": [{"messages": ["c0"]}],
    }
    config = TrainDatasetConfig(
        sources=[
            DatasetSourceConfig(path="math", type="rl", teacher_group="math_group"),
            DatasetSourceConfig(path="code", type="rl", teacher_group="code_group"),
        ]
    )

    dataset = get_mopd_dataset(
        config,
        source_loader=lambda **kwargs: samples[kwargs["source"].path],
    )

    assert len(dataset) == 3
    assert [dataset[index][MOPD_ROUTE_METADATA_KEY].route for index in range(3)] == [
        "math_group",
        "math_group",
        "code_group",
    ]
    assert all("task_type" not in dataset[index] for index in range(3))
    assert all(
        MOPD_ROUTE_METADATA_KEY not in sample
        for data in samples.values()
        for sample in data
    )


def test_routed_dataset_uniform_policy_balances_unequal_sources():
    """Uniform policy deterministically cycles shorter sources per epoch."""
    config = TrainDatasetConfig(
        mixture_sampling_policy="uniform",
        sources=[
            DatasetSourceConfig(path="short", type="rl", teacher_group="short-group"),
            DatasetSourceConfig(path="long", type="rl", teacher_group="long-group"),
        ],
    )
    samples = {
        "short": [{"id": "s0"}],
        "long": [{"id": "l0"}, {"id": "l1"}, {"id": "l2"}],
    }

    dataset = get_mopd_dataset(
        config,
        source_loader=lambda **kwargs: samples[kwargs["source"].path],
    )

    assert len(dataset) == 6
    assert [dataset[index]["id"] for index in range(6)] == [
        "s0",
        "l0",
        "s0",
        "l1",
        "s0",
        "l2",
    ]


@pytest.mark.parametrize("field", [MOPD_ROUTE_METADATA_KEY, "mopd_route"])
def test_mopd_dataset_rejects_sample_level_route(field):
    """Samples cannot override the route declared by their source."""
    dataset = MOPDDataset([([{field: "sample-route"}], "source-route")])

    with pytest.raises(ValueError, match="configured only on the dataset source"):
        dataset[0]


def test_dataset_mixture_without_teacher_group_has_no_route_metadata():
    """Non-MOPD mixtures leave samples free of teacher-routing metadata."""
    config = TrainDatasetConfig(
        sources=[DatasetSourceConfig(path="general", type="rl")]
    )
    samples = [{"id": "sample"}]

    dataset = get_mopd_dataset(
        config,
        source_loader=lambda **kwargs: samples,
    )

    assert dataset[0] == {"id": "sample"}
    assert MOPD_ROUTE_METADATA_KEY not in samples[0]


class _RemoteSource(RDataset):
    def __init__(self, samples):
        self.samples = samples
        self.connect_calls = []
        self.prefetch_indices = []
        self.closed = False

    def connect(self, controller, dataset_id, **kwargs):
        self.connect_calls.append((controller, dataset_id, kwargs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def _start_prefetch(self, indices):
        self.prefetch_indices.append(indices)

    def close(self):
        self.closed = True


def test_remote_mopd_dataset_connects_and_prefetches_each_source():
    """Global mixture indices are translated to each remote source correctly."""
    first = _RemoteSource([{"id": 0}, {"id": 1}])
    second = _RemoteSource([{"id": 2}, {"id": 3}, {"id": 4}])
    dataset = MOPDDataset([(first, "r0"), (second, "r1")])

    assert is_remote_dataset(dataset)
    dataset.connect(
        "controller",
        dataset_id="mixture",
        tokenizer_or_processor_path="tokenizer",
        shuffle=True,
        drop_last=True,
    )
    dataset._start_prefetch([4, 0, 2, 1, 3])

    assert first.connect_calls[0][1] == "mixture_source_0"
    assert second.connect_calls[0][1] == "mixture_source_1"
    assert first.prefetch_indices == [[0, 1]]
    assert second.prefetch_indices == [[2, 0, 1]]
    assert dataset[2][MOPD_ROUTE_METADATA_KEY].route == "r1"

    dataset.close()
    assert first.closed and second.closed
