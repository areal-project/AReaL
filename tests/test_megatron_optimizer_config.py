# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from areal.api import FinetuneSpec
from areal.api.cli_args import MegatronEngineConfig, OptimizerConfig
from areal.engine import megatron_engine as megatron_engine_module
from areal.engine.core.train_engine import _GradientAccumulationState
from areal.infra.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.npu,
    pytest.mark.skipif(
        current_platform.device_type not in ("cuda", "npu")
        or current_platform.device_count() < 1,
        reason="Requires one GPU or Ascend NPU",
    ),
]
MegatronEngine = megatron_engine_module.MegatronEngine


@pytest.mark.parametrize("last", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("precision", ["bf16", "fp8", "fp4", "kitchen"])
def test_accumulation_schedule_restores_callbacks(monkeypatch, last, fail, precision):
    """Preserve accumulated gradients and restore callbacks and TE cache flags."""
    from megatron.core.extensions import transformer_engine as te_ext

    monkeypatch.setattr(megatron_engine_module, "get_model_config", lambda m: m.config)
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.mcore_config = SimpleNamespace(enable_mtp_training=False)
    finalized = []
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(
        no_sync_func=object(),
        grad_sync_func=object(),
        param_sync_func=object(),
        finalize_model_grads_func=lambda *a, **kw: finalized.append((a, kw)),
        fp8="hybrid" if precision == "fp8" else None,
        fp4="nvfp4" if precision == "fp4" else None,
        use_kitchen=precision == "kitchen",
    )
    wrappers = []
    for name, original_cache in zip(
        ("TELinear", "TELayerNormColumnParallelLinear", "TEGroupedLinear"),
        (False, True, False),
        strict=True,
    ):
        wrapper_type = type(name, (torch.nn.Module,), {})
        monkeypatch.setattr(te_ext, name, wrapper_type)
        wrapper = wrapper_type()
        wrapper.disable_parameter_transpose_cache = original_cache
        # A quantized model can contain BF16 wrappers reset by the same schedule.
        wrapper.config = SimpleNamespace(fp8=None, fp4=None, use_kitchen=False)
        model.add_module(name, wrapper)
        wrappers.append((wrapper, original_cache))
    unrelated_module = torch.nn.Module()
    unrelated_module.disable_parameter_transpose_cache = False
    model.add_module("unrelated", unrelated_module)
    original = vars(model.config).copy()
    model.ddp_config = SimpleNamespace(overlap_grad_reduce=True)
    for parameter in model.parameters():
        parameter.main_grad = torch.ones_like(parameter)
    engine.model = [model]
    state = _GradientAccumulationState(2, next_index=int(last))
    state.add_chunk(torch.tensor(100.0), 2)
    if last:
        state.add_chunk(torch.tensor(300.0), 3)
    try:
        with engine._accumulation_schedule_context(state, False):
            assert not model.ddp_config.overlap_grad_reduce
            assert model.config.no_sync_func is original["no_sync_func"]
            assert model.config.grad_sync_func is None
            assert model.config.param_sync_func is (
                None if last else original["param_sync_func"]
            )
            for wrapper, original_cache in wrappers:
                assert wrapper.disable_parameter_transpose_cache is (
                    True if precision != "bf16" else original_cache
                )
            assert not unrelated_module.disable_parameter_transpose_cache
            if fail:
                raise RuntimeError("schedule failed")
            model.config.finalize_model_grads_func(engine.model, pg_collection="groups")
            assert len(finalized) == int(last)
            for parameter in model.parameters():
                torch.testing.assert_close(
                    parameter.main_grad,
                    torch.full_like(parameter, 0.5 if last else 1.0),
                    rtol=0,
                    atol=0,
                )
    except RuntimeError as error:
        assert fail and str(error) == "schedule failed"
    assert vars(model.config) == original
    assert model.ddp_config.overlap_grad_reduce
    for wrapper, original_cache in wrappers:
        assert wrapper.disable_parameter_transpose_cache is original_cache
    assert not unrelated_module.disable_parameter_transpose_cache


@pytest.mark.parametrize(
    "wrapper_name", ["TELinear", "TELayerNormColumnParallelLinear", "TEGroupedLinear"]
)
def test_accumulation_wrapper_passes_no_first_microbatch_to_backend(
    monkeypatch, wrapper_name
):
    """Exercise the real MCore wrapper forward while substituting device kernels."""
    from megatron.core.extensions import transformer_engine as te_ext

    received = []

    def backend_forward(self, inputs, *args, is_first_microbatch=None):
        received.append(is_first_microbatch)
        return inputs

    wrapper_type = getattr(te_ext, wrapper_name)
    monkeypatch.setattr(wrapper_type.__mro__[1], "forward", backend_forward)
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.disable_parameter_transpose_cache = False
    wrapper.te_quant_params = None
    wrapper.te_return_bias = False
    model = torch.nn.Module()
    model.layer = wrapper
    model.config = SimpleNamespace(
        fp8="hybrid", finalize_model_grads_func=lambda *a: None
    )
    model.ddp_config = SimpleNamespace(overlap_grad_reduce=True)
    engine = MegatronEngine.__new__(MegatronEngine)
    engine.model = [model]
    engine.mcore_config = SimpleNamespace(enable_mtp_training=False)
    monkeypatch.setattr(megatron_engine_module, "get_model_config", lambda m: m.config)
    inputs = torch.ones(1, 1)
    args = (inputs, [1]) if wrapper_name == "TEGroupedLinear" else (inputs,)
    state = _GradientAccumulationState(2, next_index=1)

    with engine._accumulation_schedule_context(state, False):
        # Each new schedule can reset the wrapper's flag, including later chunks.
        wrapper.is_first_microbatch = True
        wrapper(*args)
        assert received == [None]

    assert not wrapper.disable_parameter_transpose_cache
    wrapper.is_first_microbatch = True
    wrapper(*args)
    assert received == [None, True]


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
    engine.lora_mode = False
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


def test_mark_duplicated_params_clears_tp_metadata_for_replicated_params():
    model = torch.nn.Module()

    duplicated_linear = torch.nn.Linear(2, 2, bias=False)
    duplicated_linear.tp_size = 1
    duplicated_linear.weight.tensor_model_parallel = True

    expert_linear = torch.nn.Linear(2, 2, bias=False)
    expert_linear.tp_size = 1
    expert_linear.weight.tensor_model_parallel = True

    sharded_linear = torch.nn.Linear(2, 2, bias=False)
    sharded_linear.tp_size = 4
    sharded_linear.weight.tensor_model_parallel = True

    model.add_module("duplicated_linear", duplicated_linear)
    model.add_module("moe_expert_linear", expert_linear)
    model.add_module("sharded_linear", sharded_linear)

    engine = type("DummyEngine", (), {"model": [model]})()

    MegatronEngine._mark_duplicated_params(engine)

    assert duplicated_linear.weight._is_duplicated
    assert not duplicated_linear.weight.tensor_model_parallel

    assert not hasattr(expert_linear.weight, "_is_duplicated")
    assert expert_linear.weight.tensor_model_parallel

    assert not hasattr(sharded_linear.weight, "_is_duplicated")
    assert sharded_linear.weight.tensor_model_parallel


@pytest.mark.parametrize(
    "routing_type,coefficient,z_loss,error",
    [
        ("aux_loss", 0.01, None, "load-balancing"),
        (["aux_loss", "seq_aux_loss"], [0.0, 0.01], None, "load-balancing"),
        ("aux_loss", 0.0, 1e-5, "moe_z_loss_coeff"),
        ("aux_loss", 0.0, None, None),
        ("none", 0.01, 0.0, None),
    ],
)
def test_accumulation_rejects_enabled_router_losses_without_overwriting_config(
    routing_type, coefficient, z_loss, error
):
    from megatron.core.transformer.moe.router import TopKRouter

    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.config = SimpleNamespace(
        moe_aux_loss_coeff=coefficient, moe_z_loss_coeff=z_loss
    )
    router.routing_type = routing_type
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    model = torch.nn.Module()
    model.router = router
    engine.model = [model]
    engine.mcore_config = SimpleNamespace(enable_mtp_training=False)
    original = vars(router.config).copy()
    if error:
        with pytest.raises(ValueError, match=error):
            engine._validate_accumulation_model()
    else:
        engine._validate_accumulation_model()
    assert vars(router.config) == original
    assert router.routing_type == routing_type


@pytest.mark.parametrize(
    "local_error,peer_error",
    [(None, None), ("local router loss", None), (None, "peer MTP alias")],
)
def test_accumulation_initialization_checks_all_actor_ranks(
    monkeypatch, local_error, peer_error
):
    """A PP rank without the offending layer must fail before HF loading too."""
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    engine._cpu_group = object()
    engine.process_group_initialized = True

    def validate():
        if local_error is not None:
            raise ValueError(local_error)

    def gather(errors, error, *, group):
        assert group is engine.cpu_group
        assert error == local_error
        errors[:] = [error, peer_error]

    engine._validate_accumulation_model = validate
    monkeypatch.setattr(
        megatron_engine_module.dist, "get_world_size", lambda *, group: 2
    )
    monkeypatch.setattr(megatron_engine_module.dist, "all_gather_object", gather)
    error = local_error or peer_error
    if error is None:
        engine._validate_accumulation_initialization()
    else:
        with pytest.raises(ValueError, match=error):
            engine._validate_accumulation_initialization()


def test_accumulation_rejects_mtp_parameter_aliases(monkeypatch):
    engine = megatron_engine_module.MegatronEngine.__new__(
        megatron_engine_module.MegatronEngine
    )
    model = torch.nn.Module()
    model.mtp = torch.nn.Linear(2, 2)
    model.shared_weight = model.mtp.weight
    engine.model = [model]
    engine.mcore_config = SimpleNamespace(enable_mtp_training=True)
    monkeypatch.setattr(
        megatron_engine_module, "unwrap_to_gpt_model", lambda model: model
    )
    with pytest.raises(ValueError, match="isolated MTP parameters"):
        engine._accumulation_mtp_parameter_ids()
