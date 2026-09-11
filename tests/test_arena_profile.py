"""Ensure the portable Arena examples use main's configuration schema."""

from pathlib import Path

import pytest

from examples.swe.arena_config import load_arena_stream_configs
from examples.swe.utils import SWEPPOConfig

from areal.api.cli_args import load_expr_config


@pytest.mark.parametrize("profile", ["arena_grpo.yaml", "arena_multi_stream.yaml"])
def test_arena_profile_loads_with_main_schema(monkeypatch, tmp_path, profile):
    for key, value in {
        "AREAL_FILEROOT": str(tmp_path),
        "AREAL_DIR": str(Path(__file__).resolve().parents[1]),
        "AREAL_IMAGE": str(tmp_path / "image.sif"),
        "AREAL_PYTHON": "python",
        "MODEL_PATH": str(tmp_path / "model"),
        "ARENA_OPENAPI_BASE": "https://arena.example",
        "ARENA_STREAM_ID": "first",
        "SWE_RL_ADMIN_API_KEY": "test-admin",
    }.items():
        monkeypatch.setenv(key, value)

    path = Path(__file__).resolve().parents[1] / "examples/swe" / profile
    config, _ = load_expr_config(["--config", str(path)], SWEPPOConfig)
    streams = load_arena_stream_configs(config.econfig)

    assert config.cluster.n_nodes == 2
    assert config.total_train_steps == 1
    assert config.econfig.dataset_source == "arena"
    assert len(streams) == (2 if profile == "arena_multi_stream.yaml" else 1)
    assert config.rollout.agent.mode == "inline"
