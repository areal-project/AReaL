"""Focused checks for the CPU-staged Megatron DAPO-Math example."""

from __future__ import annotations

from pathlib import Path

import pytest

from examples.multi_turn_math.config import MultiTurnGRPOConfig

import areal.api.cli_args as cli_args
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import CPUStagedOffloadConfig, load_expr_config
from areal.utils.stats_logger import StatsLogger

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_cpu_staged_offload_config_validates_buffer_sizes() -> None:
    with pytest.raises(ValueError, match="buffer_count"):
        CPUStagedOffloadConfig(buffer_count=0)
    with pytest.raises(ValueError, match="bucket_size_mb"):
        CPUStagedOffloadConfig(bucket_size_mb=0)


@pytest.mark.parametrize(
    ("filename", "model_env"),
    [
        ("dapo-math_grpo_cpu_staged.yaml", "QWEN3_30B_A3B_BASE_MODEL_PATH"),
        ("dapo-math_grpo_qwen3_5_cpu_staged.yaml", "QWEN3_5_35B_A3B_BASE_MODEL_PATH"),
    ],
)
def test_example_uses_core_staged_optimizer_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, filename: str, model_env: str
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.setenv(model_env, str(tmp_path / "model"))
    monkeypatch.setattr(cli_args.name_resolve, "reconfigure", lambda config: None)
    monkeypatch.setattr(cli_args, "save_config", lambda config, path: None)
    monkeypatch.setattr(
        StatsLogger, "get_log_path", staticmethod(lambda config: str(tmp_path))
    )

    config, config_path = load_expr_config(
        [
            "--config",
            f"examples/cpu_staged_offload/{filename}",
        ],
        MultiTurnGRPOConfig,
    )
    allocation = ModelAllocation.from_str(config.actor.backend)

    assert config_path.endswith(filename)
    assert config.actor.path == config.tokenizer_path == str(tmp_path / "model")
    assert allocation.parallel.world_size == 8
    assert config.actor.megatron.cpu_staged_offload.enabled is True
    assert config.actor.megatron.cpu_staged_offload.buffer_count == 2
    assert config.actor.megatron.cpu_staged_offload.bucket_size_mb == 128
