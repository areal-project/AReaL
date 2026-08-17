"""Megatron actor and trainer used by the CPU-staged AdamW example."""

from __future__ import annotations

import os
from typing import Any

from areal import PPOTrainer
from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import PPOActorConfig
from areal.engine import MegatronPPOActor
from areal.engine.megatron_utils.gpu_staged_optimizer import GPUStagedAdamWConfig
from areal.utils.environ import is_single_controller

from .config import (
    CPU_STAGED_BUCKET_SIZE_MB_ENV,
    CPU_STAGED_BUFFER_COUNT_ENV,
    CPU_STAGED_SNAPSHOT_CHUNK_MB_ENV,
    CPU_STAGED_SNAPSHOT_ROOT_ENV,
)


def _environment_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error


def _environment_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be numeric, got {raw_value!r}") from error


def cpu_staged_adamw_config_from_environment() -> GPUStagedAdamWConfig:
    """Build the core optimizer config inside each actor worker."""
    snapshot_root = os.environ.get(CPU_STAGED_SNAPSHOT_ROOT_ENV)
    if snapshot_root is not None:
        snapshot_root = snapshot_root.strip() or None
    return GPUStagedAdamWConfig(
        buffer_count=_environment_int(CPU_STAGED_BUFFER_COUNT_ENV, 2),
        bucket_size_mb=_environment_float(CPU_STAGED_BUCKET_SIZE_MB_ENV, 128.0),
        checkpoint_snapshot_root=snapshot_root,
        checkpoint_snapshot_chunk_mb=_environment_float(
            CPU_STAGED_SNAPSHOT_CHUNK_MB_ENV, 64.0
        ),
    )


class CPUStagedMegatronPPOActor(MegatronPPOActor):
    """Megatron PPO actor that enables staged AdamW before optimizer creation."""

    def initialize(
        self,
        addr: str | None,
        ft_spec: FinetuneSpec,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.configure_gpu_staged_adamw(cpu_staged_adamw_config_from_environment())
        super().initialize(addr, ft_spec, *args, **kwargs)


CPU_STAGED_ACTOR_IMPORT_PATH = (
    "examples.cpu_staged_offload.engine.CPUStagedMegatronPPOActor"
)


class CPUStagedPPOTrainer(PPOTrainer):
    """PPO trainer that replaces only its primary Megatron actor."""

    def _create_train_engine(
        self, actor_config: PPOActorConfig, alloc: ModelAllocation
    ) -> Any:
        if alloc.name != "actor" or not self.config.cpu_staged.enabled:
            return super()._create_train_engine(actor_config, alloc)
        if alloc.backend != "megatron":
            raise ValueError(
                "the CPU-staged offload example requires actor.backend=megatron"
            )

        if is_single_controller():
            actor = CPUStagedMegatronPPOActor.as_controller(
                actor_config, self.scheduler
            )
        else:
            actor = CPUStagedMegatronPPOActor(config=actor_config)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor
