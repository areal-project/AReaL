# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from areal.api.cli_args import FP8EngineConfig, OptimizerConfig, TrainEngineConfig
from areal.engine.megatron_utils.muon import muon_config_kwargs, validate_muon_config


def test_muon_config_yaml_roundtrip_preserves_native_options():
    """Structured YAML accepts Muon and keeps its native parameter values."""
    config = OmegaConf.merge(
        OmegaConf.structured(OptimizerConfig),
        OmegaConf.create("type: muon\nmuon_momentum: 0.8\nmuon_num_ns_steps: 7"),
    )
    config = OmegaConf.to_object(config)
    options = muon_config_kwargs(config)
    assert config.type == "muon"
    assert options["muon_momentum"] == 0.8
    assert options["muon_num_ns_steps"] == 7
    assert options["muon_split_qkv"] is True
    assert options["muon_tp_mode"] == "blockwise"
    assert len(options) == 9


@pytest.mark.parametrize("kind", ["adam", "sgd", "adam_bf16"])
def test_existing_optimizers_receive_no_muon_overrides(kind):
    """Muon-specific fields do not change existing optimizer construction."""
    assert muon_config_kwargs(OptimizerConfig(type=kind)) == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("muon_scale_mode", "unknown"),
        ("muon_coefficient_type", "unknown"),
        ("muon_momentum", -0.1),
        ("muon_momentum", 1.0),
        ("muon_momentum", float("nan")),
        ("muon_num_ns_steps", 0),
        ("muon_num_ns_steps", 1.5),
        ("muon_num_ns_steps", True),
        ("muon_extra_scale_factor", float("inf")),
        ("muon_extra_scale_factor", 0),
        ("muon_tp_mode", "unknown"),
        ("muon_fp32_matmul_prec", "unknown"),
    ],
)
def test_invalid_muon_options_fail_during_config_creation(field, value):
    """Invalid numerical and enumerated options fail before model allocation."""
    with pytest.raises(ValueError, match=field):
        OptimizerConfig(type="muon", **{field: value})


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_muon_replicated_ddp_config_is_accepted(dtype):
    """The ordinary native Muon path uses DDP without state sharding."""
    config = TrainEngineConfig(dtype=dtype, optimizer=OptimizerConfig(type="muon"))
    config.megatron.ddp.use_distributed_optimizer = False
    validate_muon_config(config)


@pytest.mark.parametrize(
    "path,value,match",
    [
        ("dtype", "float16", "FP16"),
        ("use_lora", True, "LoRA"),
        ("weight_update_mode", "awex", "AWEX"),
        ("megatron.ddp.use_distributed_optimizer", True, "replicated"),
        ("megatron.ddp.overlap_param_gather", True, "parameter-gather"),
        ("megatron.overlap_param_gather_with_optimizer_step", True, "parameter-gather"),
        ("megatron.use_precision_aware_optimizer", True, "precision-aware"),
        ("megatron.fp8_config", FP8EngineConfig(), "FP8"),
        ("megatron.wrap_with_ddp", False, "DDP"),
        ("megatron.use_custom_fsdp", True, "DDP"),
        ("megatron.use_torch_fsdp2", True, "DDP"),
    ],
)
def test_unsupported_muon_combinations_fail_before_initialization(path, value, match):
    """Known unsupported combinations produce actionable errors."""
    config = TrainEngineConfig(dtype="bfloat16", optimizer=OptimizerConfig(type="muon"))
    config.megatron.ddp.use_distributed_optimizer = False
    obj = config
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    with pytest.raises(ValueError, match=match):
        validate_muon_config(config)


def test_grpo_muon_example_parses_actor_optimizer_and_dataset_slices(
    monkeypatch, tmp_path
):
    """Parse the complete GRPO example through the application's config schema."""
    from pathlib import Path

    from areal.api.cli_args import GRPOConfig, to_structured_cfg

    monkeypatch.setenv("AREAL_MUON_ADMIN_KEY", "unit-test-placeholder")
    monkeypatch.setenv("AREAL_MUON_MODEL", str(tmp_path / "model"))
    monkeypatch.setenv("AREAL_MUON_OUTPUT", str(tmp_path / "output"))
    path = (
        Path(__file__).resolve().parents[1]
        / "examples/math/gsm8k_grpo_megatron_muon.yaml"
    )
    config = OmegaConf.to_object(to_structured_cfg(OmegaConf.load(path), GRPOConfig))
    assert config.actor.optimizer.type == "muon"
    assert config.total_train_steps == 3
    assert config.train_dataset.split == "train[:24]"
    assert config.valid_dataset.split == "test[:4]"
    assert config.actor.megatron.use_checkpoint_opt_param_scheduler
    validate_muon_config(config.actor)
