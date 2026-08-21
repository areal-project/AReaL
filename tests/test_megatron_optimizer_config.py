# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api import FinetuneSpec
from areal.api.cli_args import MegatronEngineConfig, OptimizerConfig
from areal.engine import megatron_engine as megatron_engine_module
from areal.engine.megatron_utils.gpu_staged_muon import GPUStagedMuonConfig
from areal.engine.megatron_utils.gpu_staged_optimizer import GPUStagedAdamWConfig


def _make_test_engine(optimizer_config: OptimizerConfig):
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine.optimizer_config = optimizer_config
    engine.config = SimpleNamespace(use_lora=False)
    engine.mcore_config = MegatronEngineConfig()
    engine.model = [object()]
    engine.dtype = torch.bfloat16
    engine.enable_fp8 = False
    engine.fp8_config = None
    engine.process_group_initialized = True
    engine._cpu_group = object()
    return engine


def test_precision_aware_optimizer_fields_are_applied_before_validation(
    monkeypatch,
) -> None:
    """MCore must derive its precision-aware mode from the final field values."""
    captured = {}
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine.optimizer_config = OptimizerConfig(type="adam")
    engine.config = SimpleNamespace(use_lora=False)
    engine.mcore_config = MegatronEngineConfig(
        use_precision_aware_optimizer=True,
        main_grads_dtype="bfloat16",
        main_params_dtype="float32",
        exp_avg_dtype="float32",
        exp_avg_sq_dtype="float32",
    )
    engine.model = [object()]
    engine.dtype = torch.bfloat16
    engine.enable_fp8 = False
    engine.fp8_config = None

    def capture_optimizer(config, model):
        captured["config"] = config
        captured["model"] = model
        return object()

    monkeypatch.setattr(
        megatron_engine_module, "get_megatron_optimizer", capture_optimizer
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "OptimizerParamScheduler",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "MegatronCheckpointManager",
        lambda **kwargs: object(),
    )

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1)
    )

    config = captured["config"]
    assert captured["model"] is engine.model
    assert config.use_precision_aware_optimizer is True
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 is True
    assert config.main_grads_dtype is torch.bfloat16
    assert config.main_params_dtype is torch.float32
    assert config.exp_avg_dtype is torch.float32
    assert config.exp_avg_sq_dtype is torch.float32


def test_internal_gpu_staged_adamw_factory_is_explicit_and_precision_aware(
    monkeypatch, tmp_path
) -> None:
    """The opt-in factory forces required fields without changing CLI config."""
    captured = {}
    engine = _make_test_engine(OptimizerConfig(type="adam"))
    staged_config = GPUStagedAdamWConfig(
        buffer_count=3,
        bucket_size_mb=4,
        checkpoint_snapshot_root=str(tmp_path),
    )
    engine.configure_gpu_staged_adamw(staged_config)

    def capture_optimizer(config, model, config_arg):
        captured["config"] = config
        captured["model"] = model
        captured["staged_config"] = config_arg
        return object()

    monkeypatch.setattr(
        megatron_engine_module,
        "get_megatron_optimizer_with_gpu_staged_adamw",
        capture_optimizer,
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "OptimizerParamScheduler",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "MegatronCheckpointManager",
        lambda **kwargs: captured.setdefault("checkpoint", kwargs) or object(),
    )

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=2, train_batch_size=1)
    )

    config = captured["config"]
    assert captured["model"] is engine.model
    assert captured["staged_config"] is staged_config
    assert config.use_precision_aware_optimizer is True
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 is True
    assert config.main_params_dtype is torch.float32
    assert config.exp_avg_dtype is torch.float32
    assert config.exp_avg_sq_dtype is torch.float32
    assert captured["checkpoint"]["managed_checkpoint_snapshot_root"] == str(tmp_path)


def test_internal_gpu_staged_muon_uses_layerwise_factory_with_fixed_checkpoint(
    monkeypatch,
) -> None:
    """Muon opt-in wires the synchronous fixed-topology checkpoint manager."""
    captured = {}
    engine = _make_test_engine(OptimizerConfig(type="adam"))
    engine.model = None
    staged_config = GPUStagedMuonConfig(buffer_count=1, slot_size_mb=2)
    engine.configure_gpu_staged_muon(staged_config)
    assert engine.mcore_config.ddp.use_distributed_optimizer is False
    engine.model = [object()]

    def capture_optimizer(config, model, config_arg):
        captured["config"] = config
        captured["model"] = model
        captured["staged_config"] = config_arg
        return object()

    monkeypatch.setattr(
        megatron_engine_module,
        "get_megatron_optimizer_with_gpu_staged_muon",
        capture_optimizer,
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "OptimizerParamScheduler",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "MegatronCheckpointManager",
        lambda **kwargs: captured.setdefault("checkpoint", kwargs) or object(),
    )

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=2, train_batch_size=1)
    )

    config = captured["config"]
    assert captured["model"] is engine.model
    assert captured["staged_config"] is staged_config
    assert config.use_distributed_optimizer is False
    assert config.use_precision_aware_optimizer is False
    assert config.main_grads_dtype is torch.float32
    assert captured["checkpoint"]["managed_checkpoint_enabled"] is True
    assert captured["checkpoint"]["use_distributed_optimizer"] is False
    assert captured["checkpoint"]["async_save"] is False


def test_internal_gpu_staged_muon_rejects_async_checkpoint_before_mutation() -> None:
    """Muon async save is rejected before DDP or engine state changes."""
    engine = _make_test_engine(OptimizerConfig(type="adam"))
    engine.model = None
    engine.mcore_config.async_save = True
    original_use_distributed_optimizer = (
        engine.mcore_config.ddp.use_distributed_optimizer
    )

    with pytest.raises(RuntimeError, match="asynchronous checkpoint.*staged Muon"):
        engine.configure_gpu_staged_muon()

    assert (
        engine.mcore_config.ddp.use_distributed_optimizer
        is original_use_distributed_optimizer
    )
    assert getattr(engine, "_gpu_staged_muon_config", None) is None


def test_scheduler_uses_fixed_warmup_and_resume_safe_initial_lr(
    monkeypatch,
) -> None:
    captured = {}
    engine = _make_test_engine(OptimizerConfig(type="adam", warmup_steps=3))

    monkeypatch.setattr(
        megatron_engine_module,
        "get_megatron_optimizer",
        lambda config, model: object(),
    )

    def capture_scheduler(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        megatron_engine_module,
        "OptimizerParamScheduler",
        capture_scheduler,
    )
    monkeypatch.setattr(
        megatron_engine_module,
        "MegatronCheckpointManager",
        lambda **kwargs: object(),
    )

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1)
    )

    assert captured["init_lr"] == 0.0
    assert captured["lr_warmup_steps"] == 3
    assert captured["lr_decay_steps"] == 10
    assert captured["wd_incr_steps"] == 10


@pytest.mark.parametrize(
    ("optimizer_config", "ft_spec", "match"),
    [
        (
            OptimizerConfig(type="adam", warmup_steps=0),
            FinetuneSpec(total_train_epochs=0, dataset_size=10, train_batch_size=1),
            "Megatron Core OptimizerParamScheduler requires total_train_steps to be positive",
        ),
        (
            OptimizerConfig(type="adam", warmup_steps=10),
            FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1),
            "Megatron Core OptimizerParamScheduler requires warmup steps to be less than total_train_steps",
        ),
        (
            OptimizerConfig(type="adam", warmup_steps=11),
            FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1),
            "Megatron Core OptimizerParamScheduler requires warmup steps to be less than total_train_steps",
        ),
        (
            OptimizerConfig(type="adam", warmup_steps_proportion=1.0),
            FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1),
            "Megatron Core OptimizerParamScheduler requires warmup steps to be less than total_train_steps",
        ),
    ],
)
def test_scheduler_rejects_megatron_only_boundaries_before_optimizer_creation(
    monkeypatch,
    optimizer_config: OptimizerConfig,
    ft_spec: FinetuneSpec,
    match: str,
) -> None:
    engine = _make_test_engine(optimizer_config)

    def fail_if_called(*args, **kwargs):
        pytest.fail("optimizer must not be created for an invalid scheduler config")

    monkeypatch.setattr(
        megatron_engine_module,
        "get_megatron_optimizer",
        fail_if_called,
    )

    with pytest.raises(ValueError, match=match):
        engine._create_optimizer(ft_spec)
