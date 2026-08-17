"""Example-local configuration for CPU-staged AdamW.

The core AReaL CLI schema deliberately remains unchanged.  These settings are
converted to worker environment variables before the trainer creates the actor.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from areal.api.cli_args import GRPOConfig

CPU_STAGED_BUFFER_COUNT_ENV = "AREAL_CPU_STAGED_BUFFER_COUNT"
CPU_STAGED_BUCKET_SIZE_MB_ENV = "AREAL_CPU_STAGED_BUCKET_SIZE_MB"
CPU_STAGED_SNAPSHOT_ROOT_ENV = "AREAL_CPU_STAGED_SNAPSHOT_ROOT"
CPU_STAGED_SNAPSHOT_CHUNK_MB_ENV = "AREAL_CPU_STAGED_SNAPSHOT_CHUNK_MB"


@dataclass
class CPUStagedOffloadSettings:
    """Settings forwarded to the custom Megatron actor workers."""

    enabled: bool = True
    buffer_count: int = 2
    bucket_size_mb: float = 128.0
    checkpoint_snapshot_root: str | None = None
    checkpoint_snapshot_chunk_mb: float = 64.0

    def worker_environment(
        self, source: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """Return worker settings, preserving explicit environment overrides."""
        source = os.environ if source is None else source
        values = {
            CPU_STAGED_BUFFER_COUNT_ENV: source.get(
                CPU_STAGED_BUFFER_COUNT_ENV, str(self.buffer_count)
            ),
            CPU_STAGED_BUCKET_SIZE_MB_ENV: source.get(
                CPU_STAGED_BUCKET_SIZE_MB_ENV, str(self.bucket_size_mb)
            ),
            CPU_STAGED_SNAPSHOT_CHUNK_MB_ENV: source.get(
                CPU_STAGED_SNAPSHOT_CHUNK_MB_ENV,
                str(self.checkpoint_snapshot_chunk_mb),
            ),
        }
        snapshot_root = source.get(
            CPU_STAGED_SNAPSHOT_ROOT_ENV, self.checkpoint_snapshot_root
        )
        if snapshot_root is not None:
            values[CPU_STAGED_SNAPSHOT_ROOT_ENV] = snapshot_root
        return values


@dataclass
class CPUStagedGRPOConfig(GRPOConfig):
    """Multi-turn GRPO configuration with example-local staging settings."""

    agent_run_args: dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Arguments for running the multi-turn math agent."},
    )
    export_style: str = field(
        default="concat",
        metadata={"help": "How multi-turn completions are exported for training."},
    )
    cpu_staged: CPUStagedOffloadSettings = field(
        default_factory=CPUStagedOffloadSettings,
        metadata={"help": "Example-local GPU-staged AdamW settings."},
    )


def install_cpu_staged_worker_environment(
    config: CPUStagedGRPOConfig,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Expose settings to both in-process and remotely scheduled actor workers."""
    environ = os.environ if environ is None else environ
    if not config.cpu_staged.enabled:
        return {}
    values = config.cpu_staged.worker_environment(environ)
    environ.update(values)
    for task in config.actor.scheduling_spec:
        task.env_vars.update(values)
    return values
