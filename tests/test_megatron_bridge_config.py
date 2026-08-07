# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of config extraction without importing device-only backends."""

from __future__ import annotations

import ast
import dataclasses
import sys
from functools import wraps
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import (
    MegatronEngineConfig,
    MindSpeedEngineConfig,
    TrainEngineConfig,
)


def _load_function(path, name, namespace):
    source = Path(__file__).resolve().parents[1] / path
    function = next(
        node
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace[name]


@dataclasses.dataclass
class _Config:
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    num_layers: int = 5
    ffn_hidden_size: int | None = None

    def __post_init__(self):
        if self.expert_model_parallel_size <= 1:
            raise ValueError("MoE overlap requires EP > 1")
        if self.num_layers % self.pipeline_model_parallel_size:
            raise ValueError("Pipeline layout has not been configured")


@dataclasses.dataclass
class _MBridgeConfig:
    moe_token_dispatcher_type: str = "allgather"
    moe_permute_fusion: bool = False
    moe_router_fusion: bool = False
    moe_shared_expert_overlap: bool = False
    moe_router_bias_update_rate: float = 0.0
    moe_router_dtype: str | None = None
    moe_z_loss_coeff: float | None = None


@pytest.mark.parametrize("ep,etp", [(4, 1), (4, 2), (1, 1)])
def test_npu_bridge_config_validates_runtime_expert_dimensions(ep, etp):
    """Validate both configs with real EP while deferring uneven pipeline splits."""
    finalized = []

    class Provider:
        expert_model_parallel_size = 1
        expert_tensor_parallel_size = 1
        pipeline_model_parallel_size = 1
        num_layers = 5
        ffn_hidden_size = None

        def finalize(self):
            _Config(
                expert_model_parallel_size=self.expert_model_parallel_size,
                pipeline_model_parallel_size=self.pipeline_model_parallel_size,
            )
            self.ffn_hidden_size = 128
            finalized.append(self)

    class Bridge:
        hf_pretrained = SimpleNamespace(_name_or_path="")

        def to_megatron_provider(self, *, load_weights):
            assert not load_weights
            return Provider()

        @property
        def transformer_config(self):
            provider = self.to_megatron_provider(load_weights=False)
            provider.finalize()
            return provider

    extract = _load_function(
        "areal/models/mcore/registry.py",
        "make_hf_and_mcore_config",
        {
            "dataclasses": dataclasses,
            "TransformerConfig": _Config,
            "is_npu_available": True,
            "mpu": SimpleNamespace(
                get_expert_model_parallel_world_size=lambda: ep,
                get_expert_tensor_parallel_world_size=lambda: etp,
                get_pipeline_model_parallel_world_size=lambda: 4,
            ),
        },
    )
    if ep == 1:
        with pytest.raises(ValueError, match="MoE overlap requires EP > 1"):
            extract("model", None, Bridge(), "megatron-bridge")
        return

    hf_config, config = extract("model", None, Bridge(), "megatron-bridge")

    assert hf_config._name_or_path == "model"
    assert len(finalized) == 1
    assert config.expert_model_parallel_size == ep
    assert config.expert_tensor_parallel_size == etp
    assert config.pipeline_model_parallel_size == 1
    assert config.ffn_hidden_size == 128


@pytest.mark.parametrize("bridge_type", ["mbridge", "megatron-bridge"])
def test_bridge_config_existing_paths_preserve_config(bridge_type):
    """The GPU and mbridge paths keep using their original config objects."""
    config = object()
    hf_config = SimpleNamespace(_name_or_path="")
    bridge = SimpleNamespace(
        hf_pretrained=hf_config,
        hf_config=hf_config,
        config=config,
        transformer_config=config,
    )
    extract = _load_function(
        "areal/models/mcore/registry.py",
        "make_hf_and_mcore_config",
        {"is_npu_available": bridge_type == "mbridge"},
    )

    result = extract("model", None, bridge, bridge_type)

    assert result == (hf_config, config)


@pytest.mark.parametrize("pp,etp", [(1, 1), (4, 2)])
def test_mindspeed_repatch_uses_expert_group_size(monkeypatch, pp, etp):
    """MindSpeed EP excludes pipeline and expert tensor parallel dimensions."""
    captured = {}
    adaptor = ModuleType("mindspeed.megatron_adaptor")
    adaptor.repatch = captured.update
    monkeypatch.setitem(sys.modules, adaptor.__name__, adaptor)
    layout = ModuleType("areal.engine.megatron_utils.mindspeed_pipeline_layout_patch")
    layout.ensure_mindspeed_pipeline_layout_stage_count = lambda: None
    monkeypatch.setitem(sys.modules, layout.__name__, layout)
    patch = _load_function("areal/engine/megatron_engine.py", "_patch_mindspeed", {})
    strategy = ParallelStrategy(
        data_parallel_size=4,
        tensor_parallel_size=2,
        pipeline_parallel_size=pp,
        expert_parallel_size=4,
        expert_tensor_parallel_size=etp,
    )
    engine = SimpleNamespace(
        mindspeed_config=SimpleNamespace(as_dict=lambda: {}, moe_permute_fusion=False),
        mcore_config=SimpleNamespace(
            moe_permute_fusion=False,
            recompute_method="uniform",
            recompute_granularity="full",
            recompute_num_layers=1,
            ddp=SimpleNamespace(use_distributed_optimizer=True),
        ),
        bridge_cls="mbridge",
        moe_token_dispatcher_type="alltoall",
    )

    patch(engine, strategy)

    assert captured["expert_model_parallel_size"] == 4


@pytest.mark.parametrize("experts", [None, 8])
def test_qwen_vision_config_isolates_dense_moe_options(monkeypatch, experts):
    """Dense vision construction and replacement leave actor MoE flags intact."""
    calls = []

    @dataclasses.dataclass
    class VisionConfig:
        num_moe_experts: int | None = None
        moe_grouped_gemm: bool = False
        gemm_gradient_accumulation_fusion: bool = True
        moe_alltoall_overlap_comm: bool = True
        moe_permute_fusion: bool = True
        use_fused_moe_token_permute_and_unpermute: bool = True
        moe_zero_memory: str = "level0"
        moe_zero_memory_num_layers: int | None = 1

        def __post_init__(self):
            if not self.num_moe_experts:
                assert not self.gemm_gradient_accumulation_fusion
                assert not self.moe_alltoall_overlap_comm
                assert not self.moe_permute_fusion
                assert not self.use_fused_moe_token_permute_and_unpermute
                assert self.moe_zero_memory == "disable"
                assert self.moe_zero_memory_num_layers is None
            calls.append(self)

    module = ModuleType(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config"
    )
    module.Qwen3VLTransformerConfig = VisionConfig
    monkeypatch.setitem(sys.modules, module.__name__, module)
    patch = _load_function(
        "areal/engine/megatron_utils/megatron_bridge_patches.py",
        "patch_qwen3_vl_vision_config",
        {"wraps": wraps},
    )
    patch()
    installed = VisionConfig.__post_init__
    patch()
    assert VisionConfig.__post_init__ is installed

    config = VisionConfig(num_moe_experts=experts)
    clone = dataclasses.replace(config)

    assert len(calls) == 2
    assert config == clone
    assert config.gemm_gradient_accumulation_fusion == bool(experts)
    assert config.moe_alltoall_overlap_comm == bool(experts)
    assert VisionConfig.gemm_gradient_accumulation_fusion
    assert VisionConfig.moe_alltoall_overlap_comm


@pytest.mark.parametrize("is_npu", [False, True])
@pytest.mark.parametrize("aux_coeff", [0.0, 0.02, None])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("ms_fusion", [False, True])
def test_bridge_model_honors_mindspeed_aux_settings_before_finalize(
    monkeypatch, is_npu, aux_coeff, fusion, ms_fusion
):
    """NPU Bridge construction honors explicit zero and nonzero auxiliary losses."""
    provider_module = ModuleType("megatron.bridge.models.gpt_provider")
    provider_module.default_layer_spec = object()
    provider_module.local_layer_spec = object()
    monkeypatch.setitem(sys.modules, provider_module.__name__, provider_module)
    provider = SimpleNamespace(
        moe_aux_loss_coeff=0.001,
        moe_router_load_balancing_type="global_aux_loss",
        use_fused_moe_token_permute_and_unpermute=True,
    )
    finalized = []
    warning = Mock()

    def finalize():
        assert provider.moe_token_dispatcher_type == "alltoall"
        if is_npu and provider.use_fused_moe_token_permute_and_unpermute:
            provider.moe_permute_fusion = True
        expected = (ms_fusion if aux_coeff is not None else False) if is_npu else fusion
        assert provider.moe_permute_fusion is expected
        finalized.append(
            (provider.moe_aux_loss_coeff, provider.moe_router_load_balancing_type)
        )

    def provide_distributed_model(**kwargs):
        assert len(finalized) == 1
        return []

    provider.finalize = finalize
    provider.provide_distributed_model = provide_distributed_model
    make_model = _load_function(
        "areal/models/mcore/registry.py",
        "make_mcore_model",
        {
            "dataclasses": dataclasses,
            "is_npu_available": is_npu,
            "mpu": SimpleNamespace(
                get_tensor_model_parallel_world_size=lambda: 2,
                get_pipeline_model_parallel_world_size=lambda: 1,
                get_context_parallel_world_size=lambda: 8,
                get_expert_model_parallel_world_size=lambda: 8,
                get_expert_tensor_parallel_world_size=lambda: 1,
            ),
            "MCoreDDPConfig": SimpleNamespace,
            "MindSpeedEngineConfig": MindSpeedEngineConfig,
            "_configure_actor_output_layers": lambda models, config: None,
            "logger": SimpleNamespace(warning=warning),
        },
    )
    tf_config = SimpleNamespace(params_dtype="bf16", fp16=False, bf16=True)
    mindspeed_config = (
        MindSpeedEngineConfig(
            moe_aux_loss_coeff=aux_coeff,
            moe_router_load_balancing_type="aux_loss",
            moe_permute_fusion=ms_fusion,
            moe_token_dispatcher_type="alltoall" if is_npu else "allgather",
        )
        if aux_coeff is not None
        else None
    )

    make_model(
        hf_config=SimpleNamespace(model_type="test_moe"),
        tf_config=tf_config,
        mcore_config=MegatronEngineConfig(
            moe_permute_fusion=fusion, moe_token_dispatcher_type="flex"
        ),
        bridge=SimpleNamespace(to_megatron_provider=lambda **kw: provider),
        bridge_type="megatron-bridge",
        mindspeed_config=mindspeed_config,
    )

    if is_npu and mindspeed_config is not None:
        assert finalized == [(aux_coeff, "aux_loss")]
        assert tf_config.moe_aux_loss_coeff == aux_coeff
        assert tf_config.moe_router_load_balancing_type == "aux_loss"
    else:
        assert finalized == [(0.001, "global_aux_loss")]
    expected = (ms_fusion if aux_coeff is not None else False) if is_npu else fusion
    assert tf_config.moe_permute_fusion is expected
    assert tf_config.moe_token_dispatcher_type == "alltoall"
    if is_npu:
        warning.assert_not_called()
    else:
        warning.assert_called_once()
        assert "flex" in str(warning.call_args)


@pytest.mark.parametrize("ep", [1, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_initial_npu_bridge_provider_receives_fusion_before_finalize(ep, enabled):
    """Initial config extraction must override provider defaults even at EP=1."""

    @dataclasses.dataclass
    class Config:
        moe_permute_fusion: bool = False
        moe_token_dispatcher_type: str = "allgather"
        expert_model_parallel_size: int = 1
        expert_tensor_parallel_size: int = 1

    providers = []

    class Bridge:
        hf_pretrained = SimpleNamespace(_name_or_path="")

        def to_megatron_provider(self, *, load_weights):
            provider = SimpleNamespace(
                moe_permute_fusion=not enabled,
                moe_token_dispatcher_type="allgather",
                use_fused_moe_token_permute_and_unpermute=True,
            )

            def finalize():
                if provider.use_fused_moe_token_permute_and_unpermute:
                    provider.moe_permute_fusion = True
                assert provider.moe_permute_fusion is enabled
                assert provider.moe_token_dispatcher_type == "alltoall"
                assert provider.expert_model_parallel_size == ep
                providers.append(provider)

            provider.finalize = finalize
            return provider

        @property
        def transformer_config(self):
            pytest.fail("Config extraction skipped the explicit fusion override")

    extract = _load_function(
        "areal/models/mcore/registry.py",
        "make_hf_and_mcore_config",
        {
            "dataclasses": dataclasses,
            "TransformerConfig": Config,
            "MindSpeedEngineConfig": MindSpeedEngineConfig,
            "is_npu_available": True,
            "mpu": SimpleNamespace(
                get_expert_model_parallel_world_size=lambda: ep,
                get_expert_tensor_parallel_world_size=lambda: 1,
            ),
        },
    )
    bridge = Bridge()
    for _ in range(2):
        _, config = extract(
            "model",
            None,
            bridge,
            "megatron-bridge",
            moe_permute_fusion=enabled,
            moe_token_dispatcher_type="alltoall",
        )
        assert config.moe_permute_fusion is enabled
        assert config.moe_token_dispatcher_type == "alltoall"
    assert len(providers) == 2
    assert providers[0] is not providers[1]


def test_mindspeed_repatch_disables_stale_fusion_alias(monkeypatch):
    """A later engine disabling fusion must reset both cached MindSpeed flags."""
    cached = {"use_fused_moe_token_permute_and_unpermute": True}

    def repatch(updates):
        cached.update(updates)
        # MindSpeed validation promotes a stale alias back to the effective flag.
        if cached["use_fused_moe_token_permute_and_unpermute"]:
            cached["moe_permute_fusion"] = True

    adaptor = ModuleType("mindspeed.megatron_adaptor")
    adaptor.repatch = repatch
    monkeypatch.setitem(sys.modules, adaptor.__name__, adaptor)
    layout = ModuleType("areal.engine.megatron_utils.mindspeed_pipeline_layout_patch")
    layout.ensure_mindspeed_pipeline_layout_stage_count = lambda: None
    monkeypatch.setitem(sys.modules, layout.__name__, layout)
    patch = _load_function("areal/engine/megatron_engine.py", "_patch_mindspeed", {})

    for enabled in [True, False]:
        config = TrainEngineConfig(
            megatron=MegatronEngineConfig(moe_permute_fusion=not enabled),
            mindspeed=MindSpeedEngineConfig(moe_permute_fusion=enabled),
        )
        engine = SimpleNamespace(
            mindspeed_config=config.mindspeed,
            mcore_config=config.megatron,
            bridge_cls="mbridge",
            moe_token_dispatcher_type="alltoall",
        )
        patch(engine, ParallelStrategy())
        assert cached["moe_permute_fusion"] is enabled
        assert cached["use_fused_moe_token_permute_and_unpermute"] is enabled


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_npu", [False, True])
def test_mbridge_fusion_override_is_independent_of_checkpointing(
    checkpointing, enabled, is_npu
):
    """NPU mbridge receives the canonical flag before any config/model access."""
    extra = {}
    bridge = SimpleNamespace(
        set_extra_args=lambda **kwargs: extra.update(kwargs),
        TransformerConfigClass=_MBridgeConfig,
    )
    build = _load_function(
        "areal/engine/megatron_engine.py",
        "_build_hf_mcore_bridge",
        {
            "dataclasses": dataclasses,
            "_mbridge_precision_args": _load_function(
                "areal/engine/megatron_engine.py", "_mbridge_precision_args", {}
            ),
            "mbridge": SimpleNamespace(
                AutoBridge=SimpleNamespace(from_pretrained=lambda *a, **kw: bridge)
            ),
            "is_npu_available": is_npu,
        },
    )
    config = TrainEngineConfig(
        megatron=MegatronEngineConfig(
            moe_permute_fusion=not enabled, moe_token_dispatcher_type="flex"
        ),
        mindspeed=MindSpeedEngineConfig(moe_permute_fusion=enabled),
        gradient_checkpointing=checkpointing,
    )
    engine = SimpleNamespace(
        bridge_cls="mbridge",
        config=config,
        mcore_config=config.megatron,
        mindspeed_config=config.mindspeed,
        dtype="bf16",
        logger=SimpleNamespace(info=lambda *args: None),
        moe_token_dispatcher_type=(
            config.mindspeed.moe_token_dispatcher_type
            if is_npu
            else config.megatron.moe_token_dispatcher_type
        ),
    )

    assert build(engine) is bridge
    assert extra["moe_token_dispatcher_type"] == ("alltoall" if is_npu else "flex")
    if is_npu:
        assert extra["moe_permute_fusion"] is enabled
        assert extra["use_fused_moe_token_permute_and_unpermute"] is enabled
    else:
        assert extra["moe_permute_fusion"] is config.megatron.moe_permute_fusion
        assert "use_fused_moe_token_permute_and_unpermute" not in extra
    assert ("recompute_granularity" in extra) is checkpointing


@pytest.mark.parametrize("bridge_type", ["mbridge", "megatron-bridge"])
@pytest.mark.parametrize("dispatcher", ["allgather", "flex", "alltoall_seq", "invalid"])
@pytest.mark.parametrize("fusion", [False, True])
def test_npu_bridge_rejects_dispatcher_before_initialization(
    bridge_type, dispatcher, fusion
):
    """Unsupported dispatchers fail before backend lookup or process groups."""
    config = TrainEngineConfig(
        megatron=MegatronEngineConfig(bridge_type=bridge_type),
        mindspeed=MindSpeedEngineConfig(
            moe_token_dispatcher_type=dispatcher, moe_permute_fusion=fusion
        ),
    )
    initialize = _load_function(
        "areal/engine/megatron_engine.py",
        "__init__",
        {
            "torch": SimpleNamespace(bfloat16="bf16"),
            "is_npu_available": True,
            "_validate_areal_lm_head_compatibility": _load_function(
                "areal/engine/megatron_engine.py",
                "_validate_areal_lm_head_compatibility",
                {},
            ),
        },
    )
    with pytest.raises(ValueError, match="NPU requires.*alltoall"):
        initialize(SimpleNamespace(), config)


@pytest.mark.parametrize("bridge_type", ["mbridge", "megatron-bridge"])
@pytest.mark.parametrize("is_npu", [False, True])
def test_engine_resolves_dispatcher_from_platform_config(bridge_type, is_npu):
    """Conflicting platform fields have deterministic precedence."""
    config = TrainEngineConfig(
        megatron=MegatronEngineConfig(
            bridge_type=bridge_type, moe_token_dispatcher_type="flex"
        ),
        mindspeed=MindSpeedEngineConfig(),
    )
    warning = Mock()
    initialize = _load_function(
        "areal/engine/megatron_engine.py",
        "__init__",
        {
            "torch": SimpleNamespace(bfloat16="bf16"),
            "is_npu_available": is_npu,
            "_validate_areal_lm_head_compatibility": _load_function(
                "areal/engine/megatron_engine.py",
                "_validate_areal_lm_head_compatibility",
                {},
            ),
            "logger": SimpleNamespace(warning=warning),
            "importlib": SimpleNamespace(
                util=SimpleNamespace(find_spec=lambda name: True)
            ),
        },
    )
    engine = SimpleNamespace()
    initialize(engine, config)
    expected = "flex" if not is_npu and bridge_type == "mbridge" else "alltoall"
    assert engine.moe_token_dispatcher_type == expected
    if is_npu or bridge_type == "megatron-bridge":
        warning.assert_called_once()
        assert "flex" in str(warning.call_args)
    else:
        warning.assert_not_called()


@pytest.mark.parametrize("bridge_type", ["mbridge", "megatron-bridge"])
@pytest.mark.parametrize("dispatcher", ["allgather", "flex", "alltoall_seq", "invalid"])
def test_npu_dispatcher_validation_rejects_non_alltoall_without_fusion(
    bridge_type, dispatcher
):
    """Unsupported NPU dispatchers fail even when optional features are disabled."""
    config = MindSpeedEngineConfig(
        moe_token_dispatcher_type=dispatcher,
        moe_permute_fusion=False,
        moe_alltoall_overlap_comm=False,
    )
    with pytest.raises(ValueError, match="NPU requires.*alltoall"):
        config.validate_moe_token_dispatcher(bridge_type)


@pytest.mark.parametrize("bridge_type", ["mbridge", "megatron-bridge"])
@pytest.mark.parametrize("fusion", [False, True])
def test_npu_engine_warns_when_megatron_fusion_setting_is_ignored(bridge_type, fusion):
    """Only a nondefault CUDA fusion setting produces an NPU warning."""
    config = TrainEngineConfig(
        megatron=MegatronEngineConfig(
            bridge_type=bridge_type, moe_permute_fusion=fusion
        ),
        mindspeed=MindSpeedEngineConfig(moe_permute_fusion=not fusion),
    )
    warning = Mock()
    initialize = _load_function(
        "areal/engine/megatron_engine.py",
        "__init__",
        {
            "torch": SimpleNamespace(bfloat16="bf16"),
            "is_npu_available": True,
            "_validate_areal_lm_head_compatibility": _load_function(
                "areal/engine/megatron_engine.py",
                "_validate_areal_lm_head_compatibility",
                {},
            ),
            "logger": SimpleNamespace(warning=warning),
            "importlib": SimpleNamespace(
                util=SimpleNamespace(find_spec=lambda name: True)
            ),
        },
    )

    engine = SimpleNamespace()
    initialize(engine, config)

    assert engine.moe_token_dispatcher_type == "alltoall"
    if fusion:
        warning.assert_called_once()
        assert "moe_permute_fusion" in str(warning.call_args)
    else:
        warning.assert_not_called()


@pytest.mark.parametrize("dispatcher", ["allgather", "flex"])
@pytest.mark.parametrize("phase", ["initial", "model"])
def test_registry_rejects_unsupported_npu_dispatcher_before_provider(dispatcher, phase):
    """Direct registry callers receive the same validation as engine callers."""

    def unexpected_provider(**kwargs):
        pytest.fail("Provider construction ran before dispatcher validation")

    bridge = SimpleNamespace(to_megatron_provider=unexpected_provider)
    namespace = {
        "is_npu_available": True,
        "MindSpeedEngineConfig": MindSpeedEngineConfig,
    }
    name = "make_hf_and_mcore_config" if phase == "initial" else "make_mcore_model"
    function = _load_function("areal/models/mcore/registry.py", name, namespace)
    with pytest.raises(ValueError, match="NPU requires.*alltoall"):
        if phase == "initial":
            function(
                "model",
                None,
                bridge,
                "megatron-bridge",
                moe_token_dispatcher_type=dispatcher,
            )
        else:
            function(
                hf_config=None,
                tf_config=None,
                bridge=bridge,
                bridge_type="megatron-bridge",
                mindspeed_config=MindSpeedEngineConfig(
                    moe_token_dispatcher_type=dispatcher
                ),
            )


def test_explicit_overlap_remains_enabled_after_dispatcher_resolution(monkeypatch):
    """Dispatcher resolution preserves explicitly enabled overlap."""
    captured = {}
    adaptor = ModuleType("mindspeed.megatron_adaptor")
    adaptor.repatch = captured.update
    monkeypatch.setitem(sys.modules, adaptor.__name__, adaptor)
    layout = ModuleType("areal.engine.megatron_utils.mindspeed_pipeline_layout_patch")
    layout.ensure_mindspeed_pipeline_layout_stage_count = lambda: None
    monkeypatch.setitem(sys.modules, layout.__name__, layout)
    patch = _load_function("areal/engine/megatron_engine.py", "_patch_mindspeed", {})
    config = TrainEngineConfig(
        megatron=MegatronEngineConfig(moe_token_dispatcher_type="flex"),
        mindspeed=MindSpeedEngineConfig(moe_alltoall_overlap_comm=True),
    )
    engine = SimpleNamespace(
        mindspeed_config=config.mindspeed,
        mcore_config=config.megatron,
        bridge_cls="mbridge",
        moe_token_dispatcher_type=config.mindspeed.validate_moe_token_dispatcher(
            "mbridge"
        ),
    )
    patch(engine, ParallelStrategy(tensor_parallel_size=4, expert_parallel_size=4))
    assert captured["moe_token_dispatcher_type"] == "alltoall"
    assert captured["moe_alltoall_overlap_comm"] is True


@pytest.mark.parametrize("dispatcher", ["alltoall", "allgather", "flex"])
def test_cuda_initial_provider_preserves_alltoall_contract(dispatcher):
    """CUDA Megatron Bridge warns instead of activating an unqualified dispatcher."""

    @dataclasses.dataclass
    class Config:
        moe_token_dispatcher_type: str = "allgather"

    provider = SimpleNamespace(moe_token_dispatcher_type="allgather")
    warning = Mock()

    def finalize():
        assert provider.moe_token_dispatcher_type == "alltoall"

    provider.finalize = finalize
    extract = _load_function(
        "areal/models/mcore/registry.py",
        "make_hf_and_mcore_config",
        {
            "dataclasses": dataclasses,
            "TransformerConfig": Config,
            "is_npu_available": False,
            "logger": SimpleNamespace(warning=warning),
            "mpu": SimpleNamespace(
                get_expert_model_parallel_world_size=lambda: 1,
                get_expert_tensor_parallel_world_size=lambda: 1,
            ),
        },
    )
    _, config = extract(
        "model",
        None,
        SimpleNamespace(
            hf_pretrained=SimpleNamespace(), to_megatron_provider=lambda **kw: provider
        ),
        "megatron-bridge",
        moe_token_dispatcher_type=dispatcher,
    )
    assert config.moe_token_dispatcher_type == "alltoall"
    if dispatcher == "alltoall":
        warning.assert_not_called()
    else:
        warning.assert_called_once()
        assert dispatcher in str(warning.call_args)


@pytest.mark.parametrize("dispatcher", ["allgather", "flex"])
def test_cuda_mbridge_passes_selected_dispatcher(dispatcher):
    """CUDA mbridge retains its selected dispatcher instead of forcing alltoall."""
    config = TrainEngineConfig(
        megatron=MegatronEngineConfig(moe_token_dispatcher_type=dispatcher),
        mindspeed=MindSpeedEngineConfig(),
    )
    extra = {}
    bridge = SimpleNamespace(
        set_extra_args=lambda **kwargs: extra.update(kwargs),
        TransformerConfigClass=_MBridgeConfig,
    )
    build = _load_function(
        "areal/engine/megatron_engine.py",
        "_build_hf_mcore_bridge",
        {
            "dataclasses": dataclasses,
            "_mbridge_precision_args": _load_function(
                "areal/engine/megatron_engine.py", "_mbridge_precision_args", {}
            ),
            "is_npu_available": False,
            "mbridge": SimpleNamespace(
                AutoBridge=SimpleNamespace(from_pretrained=lambda *a, **kw: bridge)
            ),
        },
    )
    engine = SimpleNamespace(
        bridge_cls="mbridge",
        config=config,
        mcore_config=config.megatron,
        mindspeed_config=config.mindspeed,
        moe_token_dispatcher_type=config.megatron.moe_token_dispatcher_type,
        dtype="bf16",
        logger=SimpleNamespace(info=lambda *args: None),
    )
    build(engine)
    assert extra["moe_token_dispatcher_type"] == dispatcher
    assert extra["moe_permute_fusion"] is config.megatron.moe_permute_fusion
