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


def test_muon_algorithm_config_is_independent_from_staged_offload() -> None:
    optimizer = cli_args.OptimizerConfig(type="dist_muon")
    megatron = cli_args.MegatronEngineConfig(
        cpu_staged_offload=CPUStagedOffloadConfig(enabled=True)
    )

    assert optimizer.type == "dist_muon"
    assert optimizer.muon.momentum == 0.95
    assert megatron.cpu_staged_offload.enabled is True


def test_example_uses_core_staged_optimizer_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.setenv("QWEN3_30B_A3B_BASE_MODEL_PATH", "/models/Qwen3-30B-A3B-Base")
    monkeypatch.setattr(cli_args.name_resolve, "reconfigure", lambda config: None)
    monkeypatch.setattr(cli_args, "save_config", lambda config, path: None)
    monkeypatch.setattr(
        StatsLogger, "get_log_path", staticmethod(lambda config: str(tmp_path))
    )

    config, config_path = load_expr_config(
        [
            "--config",
            "examples/cpu_staged_offload/dapo-math_grpo_cpu_staged.yaml",
        ],
        MultiTurnGRPOConfig,
    )
    allocation = ModelAllocation.from_str(config.actor.backend)

    assert config_path.endswith("dapo-math_grpo_cpu_staged.yaml")
    assert config.actor.path == config.tokenizer_path == "/models/Qwen3-30B-A3B-Base"
    assert allocation.parallel.world_size == 8
    assert config.actor.megatron.cpu_staged_offload.enabled is True
    assert config.actor.megatron.cpu_staged_offload.buffer_count == 2
    assert config.actor.megatron.cpu_staged_offload.bucket_size_mb == 128
