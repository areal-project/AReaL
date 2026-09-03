# SPDX-License-Identifier: Apache-2.0

"""Generic routed dataset mixtures used by MOPD and other consumers."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from areal.api.cli_args import _DatasetConfig

ROUTE_METADATA_KEY = "__areal_route"
MOPD_ROUTE_METADATA_KEY = ROUTE_METADATA_KEY


@dataclass(frozen=True)
class DatasetRoute:
    """Typed route provenance stripped before a sample reaches its workflow."""

    source_index: int
    route: str

    def __post_init__(self) -> None:
        if self.source_index < 0:
            raise ValueError("source_index must be non-negative")
        if not isinstance(self.route, str) or not self.route.strip():
            raise ValueError("route must be a non-empty string")


class RoutedDataset:
    """Combine dataset sources with an explicit deterministic sampling policy."""

    def __init__(
        self,
        sources: list[tuple[Any, str | None]],
        *,
        sampling_policy: str = "proportional",
    ) -> None:
        if not sources:
            raise ValueError("Routed dataset sources must not be empty")
        if sampling_policy not in ("proportional", "uniform"):
            raise ValueError(
                "sampling_policy must be 'proportional' or 'uniform', "
                f"got {sampling_policy!r}"
            )
        self._datasets = [dataset for dataset, _ in sources]
        self._teacher_groups = [teacher_group for _, teacher_group in sources]
        self._sampling_policy = sampling_policy
        self._offsets: list[int] = []
        self._source_lengths: list[int] = []

        from areal.infra.data_service.rdataset import RDataset

        remote = [isinstance(dataset, RDataset) for dataset in self._datasets]
        if any(remote) and not all(remote):
            raise ValueError("MOPD dataset cannot mix local and remote sources")
        self._is_remote = all(remote)
        if not self._is_remote:
            self._refresh_offsets()

    @property
    def is_remote(self) -> bool:
        return self._is_remote

    def _refresh_offsets(self) -> None:
        total = 0
        offsets: list[int] = []
        source_lengths: list[int] = []
        for dataset in self._datasets:
            source_length = len(dataset)
            source_lengths.append(source_length)
            total += source_length
            offsets.append(total)
        self._offsets = offsets
        self._source_lengths = source_lengths
        if self._sampling_policy == "uniform" and any(
            length == 0 for length in source_lengths
        ):
            raise ValueError("uniform routed mixtures do not support empty sources")

    def __len__(self) -> int:
        if not self._offsets:
            self._refresh_offsets()
        if self._sampling_policy == "uniform":
            return max(self._source_lengths) * len(self._datasets)
        return self._offsets[-1]

    def _locate(self, index: int) -> tuple[int, int]:
        if self._sampling_policy == "uniform":
            source_index = index % len(self._datasets)
            local_index = (index // len(self._datasets)) % self._source_lengths[
                source_index
            ]
            return source_index, local_index
        source_index = bisect_right(self._offsets, index)
        source_start = 0 if source_index == 0 else self._offsets[source_index - 1]
        return source_index, index - source_start

    def __getitem__(self, index: int) -> dict[str, Any]:
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(index)

        source_index, local_index = self._locate(index)
        sample = self._datasets[source_index][local_index]
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"Dataset mixture samples must be mappings, got {type(sample).__name__}"
            )
        teacher_group = self._teacher_groups[source_index]
        if teacher_group is None:
            return dict(sample)
        if ROUTE_METADATA_KEY in sample or "mopd_route" in sample:
            raise ValueError(
                "teacher group must be configured only on the dataset source"
            )

        routed_sample = dict(sample)
        routed_sample[ROUTE_METADATA_KEY] = DatasetRoute(
            source_index=source_index,
            route=teacher_group,
        )
        return routed_sample

    def connect(
        self,
        controller: Any,
        dataset_id: str,
        tokenizer_or_processor_path: str = "",
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        """Connect every remote source to one shared data controller."""
        if not self._is_remote:
            raise RuntimeError("Only remote MOPD datasets require connect()")
        for index, dataset in enumerate(self._datasets):
            dataset.connect(
                controller,
                dataset_id=f"{dataset_id}_source_{index}",
                tokenizer_or_processor_path=tokenizer_or_processor_path,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        self._refresh_offsets()

    def _start_prefetch(self, indices: list[int]) -> None:
        """Translate global sampler indices to each remote source."""
        if not self._is_remote:
            return
        source_indices: list[list[int]] = [[] for _ in self._datasets]
        for index in indices:
            source_index, local_index = self._locate(index)
            source_indices[source_index].append(local_index)
        for dataset, local_indices in zip(self._datasets, source_indices, strict=True):
            dataset._start_prefetch(local_indices)

    def close(self) -> None:
        """Close all remote source proxies."""
        if not self._is_remote:
            return
        for dataset in self._datasets:
            dataset.close()


def is_remote_dataset(dataset: Any) -> bool:
    """Return whether a dataset needs data-service connection and prefetching."""
    from areal.infra.data_service.rdataset import RDataset

    return isinstance(dataset, RDataset) or (
        isinstance(dataset, RoutedDataset) and dataset.is_remote
    )


def get_routed_dataset(
    dataset_config: _DatasetConfig,
    tokenizer: Any = None,
    processor: Any = None,
    source_loader: Callable[..., Any] | None = None,
) -> RoutedDataset:
    """Load configured sources and attach optional teacher-group metadata."""
    if not dataset_config.sources:
        raise ValueError("Dataset mixture config must contain at least one source")

    if source_loader is None:
        from areal.dataset import get_custom_dataset

        def source_loader(**kwargs):
            source_config = kwargs["source_config"]
            split = kwargs["split"]
            return get_custom_dataset(
                split=split,
                dataset_config=source_config,
                tokenizer=kwargs["tokenizer"],
                processor=kwargs["processor"],
            )

    routed_sources: list[tuple[Any, str | None]] = []
    for source in dataset_config.sources:
        split = source.split or dataset_config.split
        source_config = _DatasetConfig(
            path=source.path,
            type=source.type,
            split=split,
            max_length=(
                source.max_length
                if source.max_length is not None
                else dataset_config.max_length
            ),
            dataset_kwargs=dataset_config.dataset_kwargs | source.dataset_kwargs,
            scheduling_spec=dataset_config.scheduling_spec,
        )
        dataset = source_loader(
            source=source,
            source_config=source_config,
            split=split,
            tokenizer=tokenizer,
            processor=processor,
        )
        routed_sources.append((dataset, source.teacher_group))
    return RoutedDataset(
        routed_sources,
        sampling_policy=dataset_config.mixture_sampling_policy,
    )


# Compatibility names for the first MOPD consumer of the routed mixture API.
MOPDDataset = RoutedDataset
get_mopd_dataset = get_routed_dataset


__all__ = [
    "MOPDDataset",
    "MOPD_ROUTE_METADATA_KEY",
    "ROUTE_METADATA_KEY",
    "DatasetRoute",
    "RoutedDataset",
    "get_mopd_dataset",
    "get_routed_dataset",
    "is_remote_dataset",
]
