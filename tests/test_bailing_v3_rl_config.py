# SPDX-License-Identifier: Apache-2.0

"""Validate the separated V3 NCCL example against actual config dataclasses."""

from pathlib import Path

from omegaconf import OmegaConf

from examples.swe.utils import SWEPPOConfig

from areal.api import ModelAllocation
from areal.api.cli_args import to_structured_cfg


def test_bailing_v3_recipe_uses_disjoint_nccl_allocations(monkeypatch, tmp_path):
    """Both role counts fit the cluster, and the selected path never uses AWEX."""
    for name in (
        "V3_MODEL",
        "SWE_RL_DATASET",
        "AREAL_ROOT",
        "SWE_AGENT_ROOT",
        "AREAL_SHARED_ROOT",
        "V3_ACTOR_IMAGE",
        "V3_ROLLOUT_IMAGE",
        "THETA_SGLANG_ROOT",
    ):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("AENV_SYSTEM_URL", "http://localhost:8080")
    monkeypatch.setenv("SWE_RL_ADMIN_API_KEY", "test-only-non-default-key")
    root = Path(__file__).resolve().parents[1]
    raw = OmegaConf.load(root / "examples/swe/bailing_v3_grpo.yaml")
    config = OmegaConf.to_object(to_structured_cfg(raw, SWEPPOConfig))
    actor = ModelAllocation.from_str(config.actor.backend, name="actor").parallel
    rollout = ModelAllocation.from_str(config.rollout.backend, name="rollout").parallel

    assert actor.world_size == rollout.world_size == 64
    assert config.cluster.n_nodes * config.cluster.n_gpus_per_node == (
        actor.world_size + rollout.world_size
    )
    assert config.actor.weight_update_mode == "xccl"
    assert config.actor._version == config.rollout._version == "v1"
    assert config.rollout.scheduling_strategy.type == "separation"
    assert config.enable_offload is False
    assert rollout.pp_size == config.sglang.dp_size == 1
    assert config.actor.use_lora is False
    assert config.actor.dtype == config.sglang.dtype == "bfloat16"
    assert config.sglang.enable_memory_saver is False
    assert config.rollout.agent.tool_call_parser == "ling3"
    assert config.rollout.agent.reasoning_parser == "ling3"
    assert config.rollout.scheduling_spec[0].env_vars["AREAL_SGLANG_FORK"] == "theta"
    assert (
        config.actor.scheduling_spec[0].image != config.rollout.scheduling_spec[0].image
    )
    # The two-stage training pipeline needs at least four sequences per DP rank.
    assert config.train_dataset.batch_size * config.gconfig.n_samples >= (
        actor.data_parallel_size * 2 * actor.pp_size
    )
