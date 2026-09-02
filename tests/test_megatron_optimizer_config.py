# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api import FinetuneSpec
from areal.api.cli_args import MegatronEngineConfig, OptimizerConfig
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
    return engine


def test_optimizer_loss_scale_is_wired_to_every_model_config() -> None:
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )

    def scale_loss(loss):
        return loss

    engine.optimizer = SimpleNamespace(scale_loss=scale_loss)
    config_a = SimpleNamespace(grad_scale_func=None)
    config_b = SimpleNamespace(grad_scale_func=None)
    engine.model = [
        SimpleNamespace(config=config_a),
        SimpleNamespace(module=SimpleNamespace(config=config_b)),
        SimpleNamespace(config=config_a),
    ]

    engine._set_optimizer_grad_scale_func()

    assert config_a.grad_scale_func is scale_loss
    assert config_b.grad_scale_func is scale_loss


def test_optimizer_loss_scale_is_not_wired_without_optimizer() -> None:
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    config = SimpleNamespace(grad_scale_func=None)
    engine.optimizer = None
    engine.model = [SimpleNamespace(config=config)]

    engine._set_optimizer_grad_scale_func()

    assert config.grad_scale_func is None


def test_train_batch_does_not_apply_optimizer_loss_scale_manually(
    monkeypatch,
) -> None:
    class _MicroBatchList:
        mbs = [{}, {}]

        def __len__(self):
            return len(self.mbs)

    class _Optimizer:
        def get_loss_scale(self):
            raise AssertionError("loss scale must be applied by MCore")

    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine._awex_adapter = None
    engine.device = torch.device("cpu")
    engine.optimizer = _Optimizer()
    engine._ensure_ready = lambda: None
    engine.optimizer_zero_grad = lambda: None
    engine._normalize_batch_input = lambda input_: (input_, None)
    engine._prepare_mb_list = lambda input_: _MicroBatchList()
    engine.optimizer_step = lambda: {}
    engine._collect_mtp_loss = lambda num_microbatches: None

    captured = {}

    def capture_loss(*args, loss_multiplier, **kwargs):
        captured["loss_multiplier"] = loss_multiplier
        return torch.tensor(0.0)

    engine._compute_logprobs_and_loss = capture_loss
    engine.forward_backward_batch = (
        lambda mb_list, process_output, forward_only: process_output(
            torch.tensor(0.0), {}
        )
    )

    monkeypatch.setattr(
        megatron_engine_module, "tensor_container_to", lambda value, device: value
    )
    monkeypatch.setattr(
        megatron_engine_module, "compute_total_loss_weight", lambda *args, **kwargs: 1
    )
    monkeypatch.setattr(
        megatron_engine_module.mpu,
        "get_data_parallel_group",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        megatron_engine_module.mpu,
        "get_data_parallel_world_size",
        lambda: 3,
    )

    engine.train_batch(
        input_={},
        loss_fn=lambda *args, **kwargs: torch.tensor(0.0),
        loss_weight_fn=lambda input_: torch.tensor(1),
    )

    assert captured["loss_multiplier"] == 6


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


def test_dte_records_optimizer_step_lr_before_scheduler_advances() -> None:
    """AdamW inversion must retain the LR consumed by the completed step."""
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    param_groups = [{"lr": 3e-6}, {"lr": 4e-6}]

    class _Optimizer:
        def __init__(self):
            self.param_groups = param_groups

        def step(self):
            return True, torch.tensor(1.0), None

    class _Scheduler:
        def step(self, increment):
            assert increment == 1
            param_groups[0]["lr"] = 2e-6
            param_groups[1]["lr"] = 1e-6

    engine.optimizer = _Optimizer()
    engine.lr_scheduler = _Scheduler()
    engine._dte_runtime_config = SimpleNamespace(enabled=True)

    engine.optimizer_step()
    engine.lr_scheduler_step()

    assert param_groups[0]["lr"] == 2e-6
    assert param_groups[1]["lr"] == 1e-6
    assert param_groups[0]["_areal_last_step_lr"] == 3e-6
    assert param_groups[1]["_areal_last_step_lr"] == 4e-6


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
