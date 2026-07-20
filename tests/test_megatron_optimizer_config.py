# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from areal.api import FinetuneSpec
from areal.api.cli_args import MegatronEngineConfig, OptimizerConfig
from areal.engine import megatron_engine as megatron_engine_module


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
