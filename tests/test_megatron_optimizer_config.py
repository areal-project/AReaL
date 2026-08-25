# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api import FinetuneSpec
from areal.api.cli_args import (
    CPUStagedOffloadConfig,
    DistributedDataParallelConfig,
    MegatronEngineConfig,
    MuonOptimizerConfig,
    OptimizerConfig,
)
from areal.engine import megatron_engine as megatron_engine_module


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


def test_cpu_staged_adamw_config_selects_precision_aware_factory(monkeypatch) -> None:
    """The core Megatron config directly selects the staged AdamW factory."""
    captured = {}
    engine = _make_test_engine(OptimizerConfig(type="adam"))
    engine.mcore_config.cpu_staged_offload = CPUStagedOffloadConfig(
        enabled=True, buffer_count=3, bucket_size_mb=4
    )

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
    assert captured["staged_config"].buffer_count == 3
    assert captured["staged_config"].bucket_size_mb == 4
    assert config.use_precision_aware_optimizer is True
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 is True
    assert config.main_params_dtype is torch.float32
    assert config.exp_avg_dtype is torch.float32
    assert config.exp_avg_sq_dtype is torch.float32
    assert "managed_checkpoint_snapshot_root" not in captured["checkpoint"]


def test_cpu_staged_dist_muon_selects_layerwise_factory(monkeypatch) -> None:
    """The core Megatron config directly selects the staged Muon factory."""
    captured = {}
    engine = _make_test_engine(
        OptimizerConfig(
            type="dist_muon",
            muon=MuonOptimizerConfig(momentum=0.91, num_ns_steps=4),
        )
    )
    engine.mcore_config = MegatronEngineConfig(
        ddp=DistributedDataParallelConfig(use_distributed_optimizer=False),
        cpu_staged_offload=CPUStagedOffloadConfig(
            enabled=True,
            buffer_count=1,
            bucket_size_mb=8,
        ),
    )

    class FakeMuonOptimizer:
        def bind_managed_checkpoint_process_group(self, group) -> None:
            captured["checkpoint_group"] = group

    def capture_optimizer(config, model, config_arg):
        captured["config"] = config
        captured["model"] = model
        captured["staged_config"] = config_arg
        return FakeMuonOptimizer()

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
        lambda **kwargs: object(),
    )

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=2, train_batch_size=1)
    )

    config = captured["config"]
    assert captured["model"] is engine.model
    assert captured["staged_config"].buffer_count == 1
    assert captured["staged_config"].slot_size_mb == 8
    assert config.use_distributed_optimizer is False
    assert config.use_precision_aware_optimizer is False
    assert config.main_grads_dtype is torch.float32
    assert config.optimizer == "dist_muon"
    assert config.muon_momentum == 0.91
    assert config.muon_num_ns_steps == 4
    assert captured["checkpoint_group"] is engine.cpu_group


def test_native_dist_muon_selects_official_layerwise_factory(monkeypatch) -> None:
    captured = {}
    engine = _make_test_engine(OptimizerConfig(type="dist_muon"))
    engine.mcore_config.ddp.use_distributed_optimizer = False

    def capture_optimizer(config, model):
        captured["config"] = config
        captured["model"] = model
        return object()

    monkeypatch.setattr(
        megatron_engine_module,
        "get_megatron_optimizer_with_dist_muon",
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
        lambda **kwargs: object(),
    )

    engine._create_optimizer(
        FinetuneSpec(total_train_epochs=1, dataset_size=2, train_batch_size=1)
    )

    assert captured["model"] is engine.model
    assert captured["config"].optimizer == "dist_muon"
    assert captured["config"].use_distributed_optimizer is False


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
