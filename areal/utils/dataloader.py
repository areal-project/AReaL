# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

from datasets import Dataset
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import ValidDatasetConfig, _DatasetConfig


def create_dataloader(
    dataset,
    rank: int,
    world_size: int,
    dataset_config: _DatasetConfig,
    collate_fn: Callable | None = None,
) -> StatefulDataLoader:
    """Create a stateful dataloader for a dataset with distributed sampler.

    Args:
        dataset: The dataset to create a dataloader for.
        rank: The rank of the process.
        world_size: The world size.
        dataset_config: The dataset config.
        collate_fn: The collate function to use.
    """
    if dataset_config.batch_size % world_size != 0:
        raise ValueError(
            f"batch size({dataset_config.batch_size}) must be divisible by world_size({world_size})!"
        )

    from areal.infra.data_service.rdataset import RDataset, _PrefetchAwareSampler

    drop_sampler_last = True
    if isinstance(dataset_config, ValidDatasetConfig):
        drop_sampler_last = False

    if isinstance(dataset, RDataset) and isinstance(dataset_config, ValidDatasetConfig):
        sampler_cls = _PrefetchAwareEvalSampler
    elif isinstance(dataset, RDataset):
        sampler_cls = _PrefetchAwareSampler
    elif isinstance(dataset_config, ValidDatasetConfig):
        sampler_cls = EvalDistributedSampler
    else:
        sampler_cls = DistributedSampler

    return StatefulDataLoader(
        dataset,
        batch_size=dataset_config.batch_size // world_size,
        sampler=sampler_cls(
            dataset,
            world_size,
            rank,
            shuffle=dataset_config.shuffle,
            drop_last=drop_sampler_last,
        ),
        drop_last=dataset_config.drop_last,
        num_workers=dataset_config.num_workers,
        collate_fn=collate_fn or (lambda x: x),
    )


class EvalDistributedSampler(DistributedSampler):
    r"""A DistributedSampler specifically designed for evaluation (Validation/Testing).

    Unlike the standard :class:`~torch.utils.data.DistributedSampler`, this sampler
    **does not pad** the dataset to make it evenly divisible by the number of replicas.

    In the standard implementation, extra indices are added to ensure every rank has
    the exact same `num_samples`. While useful for training (synchronized batch sizes),
    this causes some validation samples to be evaluated twice, leading to biased metrics
    (e.g., inaccurate accuracy or loss).

    **Key Behaviors:**
    1. **Exact Evaluation:** Ensures every sample in the dataset is evaluated exactly
       once across the entire cluster.
    2. **Uneven Split:** Ranks may receive different amounts of data (difference is at most 1).
       For example, if N=10 and Replicas=3, ranks get [4, 3, 3] samples respectively,
       instead of [4, 4, 4] with padding.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        if not drop_last:
            self.total_size = len(dataset)

        if self.rank + (self.num_samples - 1) * self.num_replicas >= self.total_size:
            self.num_samples -= 1


class _PrefetchAwareEvalSampler(EvalDistributedSampler):
    def __init__(self, dataset: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(dataset, *args, **kwargs)
        self._rdataset = dataset
        self._trigger_prefetch()

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._trigger_prefetch()

    def _trigger_prefetch(self) -> None:
        self._rdataset._start_prefetch(list(super().__iter__()))
