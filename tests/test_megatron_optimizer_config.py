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
    engine.bridge_cls = None
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


@pytest.mark.parametrize("per_token_loss", [False, True])
@pytest.mark.parametrize("has_token_mask", [False, True])
def test_train_batch_does_not_apply_optimizer_loss_scale_manually(
    monkeypatch,
    per_token_loss,
    has_token_mask,
) -> None:
    class _MicroBatchList:
        mbs = [
            {"loss_mask": torch.ones(1, dtype=torch.bool)} if has_token_mask else {}
            for _ in range(2)
        ]

        def __len__(self):
            return len(self.mbs)

    class _Optimizer:
        def get_loss_scale(self):
            raise AssertionError("loss scale must be applied by MCore")

    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine.model = [
        SimpleNamespace(config=SimpleNamespace(calculate_per_token_loss=per_token_loss))
    ]
    engine._awex_adapter = None
    engine._weight_residency = None
    engine.device = torch.device("cpu")
    engine.optimizer = _Optimizer()
    engine._ensure_ready = lambda: None
    engine.optimizer_zero_grad = lambda: None
    engine._normalize_batch_input = lambda input_: (input_, None)
    engine._prepare_mb_list = lambda input_, **kwargs: _MicroBatchList()
    engine.optimizer_step = lambda: {}
    engine._collect_mtp_loss = lambda num_microbatches: None

    captured = {}

    def capture_loss(*args, loss_multiplier, **kwargs):
        captured["loss_multiplier"] = loss_multiplier
        captured["per_token_loss"] = kwargs["per_token_loss"]
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

    monkeypatch.setattr(megatron_engine_module.dist, "all_reduce", lambda *a, **k: None)

    engine.train_batch(
        input_={},
        loss_fn=lambda *args, **kwargs: torch.tensor(0.0),
        loss_weight_fn=lambda input_: torch.tensor(1),
    )

    assert captured["loss_multiplier"] == (1.0 if per_token_loss else 6)
    assert captured["per_token_loss"] is per_token_loss


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
def test_per_token_normalization_preserves_policy_and_auxiliary_gradients(
    monkeypatch, cp_size, mode
):
    from areal.trainer.ppo.loss_reduction import (
        prepare_policy_gradient_batch,
    )

    mask = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.bool)
    retained = mask.clone()
    retained[0, 1] = False
    values = torch.tensor([[2.0, 8.0], [6.0, 4.0], [10.0, 0.0]], requires_grad=True)
    auxiliary = torch.tensor(2.0, requires_grad=True)
    data = {"loss_mask": mask}
    prepared = prepare_policy_gradient_batch(
        data,
        mode=mode,
        group_sizes=[2, 1],
        divisor=4.0 if mode == "constant" else None,
    )
    step = prepared.for_steps([data])[0]
    partitions = (slice(0, 1), slice(1, 3))
    batches = [{key: value[rows] for key, value in data.items()} for rows in partitions]
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine.device = torch.device("cpu")
    monkeypatch.setattr(
        megatron_engine_module.mpu, "get_context_parallel_world_size", lambda: cp_size
    )
    monkeypatch.setattr(
        megatron_engine_module.mpu, "get_data_parallel_group", lambda: None
    )
    monkeypatch.setattr(megatron_engine_module.dist, "all_reduce", lambda *a, **k: None)
    multiplier = engine._per_token_loss_multiplier(
        SimpleNamespace(mbs=batches), step.loss_weight
    )
    numerators, counts, auxiliary_numerators = [], [], []
    for rows, batch in zip(partitions, batches, strict=True):
        local_loss = step.bind(batch).aggregate(values[rows], retained[rows])
        weight = step.loss_weight(batch)
        tokens = mask[rows].count_nonzero()
        for cp_rank in range(cp_size):
            monkeypatch.setattr(
                megatron_engine_module.mpu,
                "get_context_parallel_rank",
                lambda rank=cp_rank: rank,
            )
            numerator, count = engine._build_per_token_loss_output(
                local_loss, weight, multiplier, token_count=tokens
            )
            numerators.append(numerator)
            counts.append(count)
            # MCore seeds token-summed auxiliary gradients outside the main loss.
            auxiliary_numerators.append(auxiliary * tokens / cp_size)

    total_tokens = sum(counts)
    torch.testing.assert_close(total_tokens, mask.count_nonzero(), rtol=0, atol=0)
    actual = sum(numerators) / total_tokens
    sums = torch.where(retained, values, 0).sum(-1)
    if mode == "token_mean":
        expected = sums.sum() / 5
    elif mode == "seq_mean":
        expected = (sums / torch.tensor([2, 2, 1])).mean()
    elif mode == "prompt_mean":
        expected = (sums[:2].sum() / 4 + sums[2]) / 2
    else:
        expected = sums.sum() / 12
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual_gradient = torch.autograd.grad(actual, values, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, values)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-6)
    auxiliary_loss = sum(auxiliary_numerators) / total_tokens
    torch.testing.assert_close(auxiliary_loss, auxiliary, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(auxiliary_loss, auxiliary)[0],
        torch.tensor(1.0),
        rtol=0,
        atol=0,
    )


def test_collect_mtp_loss_uses_mcore_metrics_tracker(monkeypatch) -> None:
    from megatron.core.transformer.multi_token_prediction import (
        MTPLossLoggingHelper,
    )

    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine.mcore_config = SimpleNamespace(enable_mtp_training=True)

    reduced = False
    cleaned = False

    def reduce_metrics() -> None:
        nonlocal reduced
        reduced = True

    def clean_metrics() -> None:
        nonlocal cleaned
        cleaned = True

    monkeypatch.setattr(
        MTPLossLoggingHelper,
        "tracker",
        {"loss_values": torch.tensor([2.0, 4.0])},
    )
    monkeypatch.setattr(
        MTPLossLoggingHelper,
        "reduce_metrics_in_tracker",
        reduce_metrics,
        raising=False,
    )
    monkeypatch.setattr(
        MTPLossLoggingHelper,
        "clean_metrics_in_tracker",
        clean_metrics,
        raising=False,
    )

    assert engine._collect_mtp_loss(num_microbatches=2) == 3.0
    assert reduced
    assert cleaned


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
    engine.bridge_cls = None
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
