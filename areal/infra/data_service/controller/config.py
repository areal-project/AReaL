# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import SchedulingSpec, SchedulingStrategy


@dataclass
class DataServiceConfig:
    """Internal config for the data service controller.

    Constructed from ``_DatasetConfig`` fields by the trainer.
    Not exposed to end users directly.
    """

    num_workers: int = 1
    scheduling_spec: SchedulingSpec = field(
        default_factory=lambda: SchedulingSpec(
            cpu=1, gpu=0, mem=10, cmd="python3 -m areal.infra.rpc.guard"
        ),
    )
    # Always separation — data controller starts before engines.
    scheduling_strategy: SchedulingStrategy = field(
        default_factory=lambda: SchedulingStrategy(type="separation"),
    )
    setup_timeout: float = 120.0
    dataloader_num_workers: int = 4
    seed: int = 42

    @staticmethod
    def from_dataset_config(dataset_config, seed: int = 42) -> DataServiceConfig:
        """Build from a ``_DatasetConfig`` instance."""
        return DataServiceConfig(
            num_workers=max(dataset_config.num_dataset_workers, 1),
            scheduling_spec=dataset_config.scheduling_spec,
            dataloader_num_workers=max(dataset_config.num_workers, 0),
            setup_timeout=dataset_config.setup_timeout,
            seed=seed,
        )


__all__ = ["DataServiceConfig"]
