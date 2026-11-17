# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import MegatronEngineConfig


def _load_mcore_test_modules():
    pytest.importorskip("megatron.core")
    pytest.importorskip("mbridge")
    from areal.engine.megatron_engine import MegatronEngine
    from areal.models.mcore import hf_load, registry

    return MegatronEngine, hf_load, registry


class _FakeGPTModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int = 4,
        vocab_size: int = 8,
        *,
        post_process: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ) -> None:
        super().__init__()
        self.post_process = post_process
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.embedding = torch.nn.Module()
        self.embedding.word_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self.backbone = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        if post_process:
            self.output_layer = torch.nn.Linear(hidden_size, vocab_size, bias=False)
            if share_embeddings_and_output_weights:
                self.output_layer.weight = self.embedding.word_embeddings.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(self.embedding.word_embeddings(input_ids))
        out = self.output_layer(hidden)
        return out[0] if isinstance(out, tuple) else out


class _FakeMutableVLMModel(torch.nn.Module):
    def __init__(self, language_model: _FakeGPTModel) -> None:
        super().__init__()
        self.language_model = language_model
        self.share_embeddings_and_output_weights = (
            language_model.share_embeddings_and_output_weights
        )


class _FakeReadOnlyVLMModel(torch.nn.Module):
    def __init__(self, language_model: _FakeGPTModel) -> None:
        super().__init__()
        self.language_model = language_model

    @property
    def share_embeddings_and_output_weights(self) -> bool:
        return self.language_model.share_embeddings_and_output_weights


class _FakeDDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, **_: object) -> None:
        super().__init__()
        self.module = module
        self._ddp_params: list[torch.nn.Parameter] = list(module.parameters())
        for param in self._ddp_params:
            param.main_grad = torch.zeros_like(param)
            if param.requires_grad:

                def _accumulate(grad_param: torch.nn.Parameter) -> None:
                    if grad_param.grad is not None:
                        grad_param.main_grad.add_(grad_param.grad.detach())

                param.register_post_accumulate_grad_hook(_accumulate)

    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        return self.module(*args, **kwargs)


def test_value_head_zero_initializes_weights_and_bias() -> None:
    _, _, registry = _load_mcore_test_modules()
    tf_config = SimpleNamespace(sequence_parallel=True)

    head = registry.ValueHead(input_size=6, output_size=1, config=tf_config, bias=True)

    assert head.weight.shape == (1, 6)
    assert getattr(head.weight, "sequence_parallel", False) is True
    torch.testing.assert_close(head.weight, torch.zeros(1, 6), rtol=0, atol=0)
    torch.testing.assert_close(head.bias, torch.zeros(1), rtol=0, atol=0)


def test_mbridge_critic_installs_value_head_before_ddp_and_updates_weights(
    monkeypatch,
) -> None:
    _, _, registry = _load_mcore_test_modules()
    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(registry, "DDP", _FakeDDP)

    hf_config = SimpleNamespace(tie_word_embeddings=True)
    tf_config = SimpleNamespace(
        hidden_size=4,
        params_dtype=torch.float32,
        fp16=False,
        bf16=False,
        sequence_parallel=False,
        mtp_num_layers=None,
    )
    observed: dict[str, object] = {}

    class _FakeBridge:
        def __init__(self) -> None:
            self.hf_config = hf_config
            self.config = tf_config

        def get_model(
            self,
            *,
            wrap_with_ddp: bool = True,
            ddp_config: dict | None = None,
            use_torch_fsdp2: bool = False,
            use_custom_fsdp: bool = False,
            fp16: bool = False,
            bf16: bool = False,
            use_precision_aware_optimizer: bool = False,
            overlap_param_gather_with_optimizer_step: bool = False,
            post_model_creation_callbacks: tuple = (),
        ):
            observed["tie_inside_get_model"] = self.hf_config.tie_word_embeddings
            model = _FakeGPTModel(
                hidden_size=4,
                vocab_size=8,
                post_process=True,
                share_embeddings_and_output_weights=self.hf_config.tie_word_embeddings,
            )
            for callback in post_model_creation_callbacks:
                callback(model)
            return [_FakeDDP(model)] if wrap_with_ddp else [model]

    bridge = _FakeBridge()
    models = registry.make_mcore_model(
        hf_config=hf_config,
        tf_config=tf_config,
        mcore_config=MegatronEngineConfig(bridge_type="mbridge", wrap_with_ddp=True),
        bridge=bridge,
        bridge_type="mbridge",
        is_critic=True,
    )

    assert observed["tie_inside_get_model"] is False
    assert hf_config.tie_word_embeddings is True

    ddp_model = models[0]
    gpt_model = registry.unwrap_to_gpt_model(ddp_model)
    assert gpt_model.share_embeddings_and_output_weights is False
    assert isinstance(gpt_model.output_layer, registry.ValueHead)
    assert (
        gpt_model.output_layer.weight is not gpt_model.embedding.word_embeddings.weight
    )
    assert gpt_model.output_layer.weight in set(ddp_model._ddp_params)
    assert hasattr(gpt_model.output_layer.weight, "main_grad")

    values = ddp_model(torch.tensor([[1, 2]]))
    loss = (values - 1.0).pow(2).mean()
    loss.backward()
    assert gpt_model.output_layer.weight.main_grad.abs().sum().item() > 0

    optimizer = torch.optim.SGD(ddp_model._ddp_params, lr=0.5)
    for param in ddp_model._ddp_params:
        param.grad = param.main_grad
    optimizer.step()
    assert not torch.equal(gpt_model.output_layer.weight, torch.zeros(1, 4))


@pytest.mark.parametrize("vlm_cls", [_FakeMutableVLMModel, _FakeReadOnlyVLMModel])
def test_mbridge_critic_pipeline_parallel_stages_and_vlm_wrapper_untying(
    monkeypatch,
    vlm_cls,
) -> None:
    _, _, registry = _load_mcore_test_modules()
    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(registry, "DDP", _FakeDDP)

    hf_config = SimpleNamespace(
        tie_word_embeddings=True,
        text_config=SimpleNamespace(tie_word_embeddings=True),
    )
    tf_config = SimpleNamespace(
        hidden_size=4,
        params_dtype=torch.float32,
        fp16=False,
        bf16=False,
        sequence_parallel=False,
        mtp_num_layers=None,
    )

    def _build_stage(post_process: bool):
        class _StageBridge:
            def __init__(self) -> None:
                self.hf_config = hf_config
                self.config = tf_config

            def get_model(
                self,
                *,
                wrap_with_ddp: bool = True,
                ddp_config: dict | None = None,
                use_torch_fsdp2: bool = False,
                use_custom_fsdp: bool = False,
                fp16: bool = False,
                bf16: bool = False,
                use_precision_aware_optimizer: bool = False,
                overlap_param_gather_with_optimizer_step: bool = False,
                post_model_creation_callbacks: tuple = (),
            ):
                inner = _FakeGPTModel(
                    hidden_size=4,
                    vocab_size=8,
                    post_process=post_process,
                    share_embeddings_and_output_weights=True,
                )
                emb_weight = inner.embedding.word_embeddings.weight
                emb_weight.shared_embedding = True
                emb_weight.shared = True
                emb_weight.zero_out_wgrad = True
                wrapped_vlm = vlm_cls(inner)
                for callback in post_model_creation_callbacks:
                    callback(wrapped_vlm)
                return [_FakeDDP(wrapped_vlm)]

        return registry.make_mcore_model(
            hf_config=hf_config,
            tf_config=tf_config,
            mcore_config=MegatronEngineConfig(
                bridge_type="mbridge", wrap_with_ddp=True
            ),
            bridge=_StageBridge(),
            bridge_type="mbridge",
            is_critic=True,
        )[0]

    stage_first = _build_stage(post_process=False)
    gpt_first = registry.unwrap_to_gpt_model(stage_first)
    assert not hasattr(gpt_first, "output_layer")
    assert stage_first.module.share_embeddings_and_output_weights is False
    assert gpt_first.share_embeddings_and_output_weights is False
    emb_weight = gpt_first.embedding.word_embeddings.weight
    assert emb_weight.shared_embedding is False
    assert emb_weight.shared is False
    assert emb_weight.zero_out_wgrad is False

    stage_last = _build_stage(post_process=True)
    gpt_last = registry.unwrap_to_gpt_model(stage_last)
    assert isinstance(gpt_last.output_layer, registry.ValueHead)
    assert hasattr(gpt_last.output_layer.weight, "main_grad")


def test_megatron_bridge_critic_installs_value_head_in_pre_wrap_hook(
    monkeypatch,
) -> None:
    _, _, registry = _load_mcore_test_modules()
    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(registry, "DDP", _FakeDDP)
    for getter in (
        "get_tensor_model_parallel_world_size",
        "get_pipeline_model_parallel_world_size",
        "get_context_parallel_world_size",
        "get_expert_model_parallel_world_size",
        "get_expert_tensor_parallel_world_size",
    ):
        monkeypatch.setattr(registry.mpu, getter, lambda: 1)

    events: list[tuple[str, bool]] = []

    class _FakeProvider:
        mtp_num_layers = None
        gradient_accumulation_fusion = False
        share_embeddings_and_output_weights = True

        def __init__(self) -> None:
            self.hooks: list = []

        def finalize(self) -> None:
            events.append(("finalize", self.share_embeddings_and_output_weights))

        def register_pre_wrap_hook(self, hook) -> None:
            self.hooks.append(hook)

        def provide_distributed_model(self, **kwargs):
            model = _FakeGPTModel(
                hidden_size=4,
                vocab_size=8,
                post_process=True,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            )
            models = [model]
            for hook in self.hooks:
                models = hook(models)
            events.append(
                ("pre_ddp", isinstance(model.output_layer, registry.ValueHead))
            )
            return [_FakeDDP(m) for m in models]

    bridge = SimpleNamespace(to_megatron_provider=lambda **kwargs: _FakeProvider())
    models = registry.make_mcore_model(
        hf_config=SimpleNamespace(model_type="qwen3"),
        tf_config=SimpleNamespace(
            hidden_size=4,
            params_dtype=torch.float32,
            fp16=False,
            bf16=False,
            sequence_parallel=False,
        ),
        mcore_config=MegatronEngineConfig(bridge_type="megatron-bridge"),
        bridge=bridge,
        bridge_type="megatron-bridge",
        is_critic=True,
    )

    assert events == [("finalize", False), ("pre_ddp", True)]
    gpt_model = registry.unwrap_to_gpt_model(models[0])
    assert isinstance(gpt_model.output_layer, registry.ValueHead)
    assert hasattr(gpt_model.output_layer.weight, "main_grad")


def test_verify_critic_value_heads_raises_when_installed_after_ddp(
    monkeypatch,
) -> None:
    _, _, registry = _load_mcore_test_modules()
    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(registry, "DDP", _FakeDDP)

    tf_config = SimpleNamespace(hidden_size=4, params_dtype=torch.float32)
    ddp_model = _FakeDDP(
        _FakeGPTModel(hidden_size=4, vocab_size=8, post_process=True),
    )
    registry._replace_output_layer_with_value_head(ddp_model.module, tf_config)

    with pytest.raises(RuntimeError, match="not registered in Megatron DDP"):
        registry._verify_critic_value_heads([ddp_model])


def test_critic_rejects_enable_mtp_and_drops_disabled_checkpoint_mtp(
    monkeypatch,
) -> None:
    _, _, registry = _load_mcore_test_modules()
    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(registry, "DDP", _FakeDDP)

    with pytest.raises(ValueError, match="do not support MTP"):
        registry.make_mcore_model(
            hf_config=SimpleNamespace(),
            tf_config=SimpleNamespace(hidden_size=4, params_dtype=torch.float32),
            mcore_config=MegatronEngineConfig(bridge_type="mbridge", enable_mtp=True),
            bridge=object(),
            bridge_type="mbridge",
            is_critic=True,
        )

    tf_config = SimpleNamespace(
        hidden_size=4,
        params_dtype=torch.float32,
        fp16=False,
        bf16=False,
        sequence_parallel=False,
        mtp_num_layers=1,
    )

    class _FakeBridge:
        def __init__(self) -> None:
            self.hf_config = SimpleNamespace(tie_word_embeddings=False)
            self.config = SimpleNamespace(mtp_num_layers=1)

        def get_model(
            self,
            *,
            wrap_with_ddp: bool = True,
            ddp_config: dict | None = None,
            use_torch_fsdp2: bool = False,
            use_custom_fsdp: bool = False,
            fp16: bool = False,
            bf16: bool = False,
            use_precision_aware_optimizer: bool = False,
            overlap_param_gather_with_optimizer_step: bool = False,
            post_model_creation_callbacks: tuple = (),
        ):
            model = _FakeGPTModel(hidden_size=4, vocab_size=8, post_process=True)
            for callback in post_model_creation_callbacks:
                callback(model)
            return [_FakeDDP(model)]

    bridge = _FakeBridge()
    registry.make_mcore_model(
        hf_config=bridge.hf_config,
        tf_config=tf_config,
        mcore_config=MegatronEngineConfig(bridge_type="mbridge", enable_mtp=False),
        bridge=bridge,
        bridge_type="mbridge",
        is_critic=True,
    )
    assert tf_config.mtp_num_layers is None
    assert bridge.config.mtp_num_layers is None


@pytest.mark.parametrize("has_saved_head", [True, False])
def test_megatron_engine_qwen35_mbridge_critic_uses_native_load_and_restores_value_head(
    monkeypatch,
    tmp_path,
    has_saved_head: bool,
) -> None:
    MegatronEngine, hf_load, registry = _load_mcore_test_modules()
    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(hf_load.mpu, "is_pipeline_last_stage", lambda: True)

    tf_config = SimpleNamespace(
        hidden_size=4, params_dtype=torch.float32, sequence_parallel=False
    )
    model = _FakeGPTModel(hidden_size=4, vocab_size=8, post_process=True)
    registry._replace_output_layer_with_value_head(model, tf_config)

    if has_saved_head:
        expected_weight = torch.full((1, 4), 2.5)
        torch.save({"weight": expected_weight}, tmp_path / "value_head.pt")
    else:
        expected_weight = torch.zeros(1, 4)

    native_calls: list[str] = []
    fast_calls: list[str] = []

    class _FakeBridge:
        def _get_actual_hf_path(self, path: str) -> str:
            return path

        def load_weights(self, models, path: str) -> None:
            native_calls.append(path)

    monkeypatch.setattr(
        "areal.engine.megatron_engine.load_weights_from_hf_with_mbridge_fast",
        lambda **kwargs: fast_calls.append(kwargs["weights_path"]),
    )

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.bridge_cls = "mbridge"
    engine.bridge = _FakeBridge()
    engine.model = [model]
    engine.fp8_direct_convert = False
    engine.hf_config = SimpleNamespace(model_type="qwen3_5")
    engine.config = SimpleNamespace(is_critic=True)

    engine._load_model_from_hf(str(tmp_path))

    assert native_calls == [str(tmp_path)]
    assert fast_calls == []
    torch.testing.assert_close(
        model.output_layer.weight, expected_weight, rtol=0, atol=0
    )

    # Non-Qwen3.5 mbridge models continue using the fast HF loader.
    engine.hf_config = SimpleNamespace(model_type="qwen3")
    engine._load_model_from_hf(str(tmp_path))
    assert fast_calls == [str(tmp_path)]


def test_qwen35_mbridge_load_dequantizes_fp8_weights_and_mixed_gdn_layers(
    monkeypatch,
    tmp_path,
) -> None:
    MegatronEngine, hf_load, registry = _load_mcore_test_modules()
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from mbridge.core.safetensor_io import SafeTensorIO

    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(hf_load.mpu, "is_pipeline_last_stage", lambda: True)

    shard_name = "model-00001-of-00001.safetensors"
    down_q = torch.tensor(
        [[4.0, 4.0, 8.0, 8.0], [4.0, 4.0, 8.0, 8.0]], dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    down_scale = torch.tensor([[0.01, 0.25]], dtype=torch.float32)

    in_qkv_q = torch.full((2, 4), 10.0, dtype=torch.float32).to(torch.float8_e4m3fn)
    in_qkv_scale = torch.tensor([[0.02, 0.05]], dtype=torch.float32)
    in_b_bf16 = torch.full((1, 4), 0.5, dtype=torch.bfloat16)
    lm_head_bf16 = torch.ones((8, 4), dtype=torch.bfloat16)

    tensors = {
        "model.layers.0.mlp.down_proj.weight": down_q,
        "model.layers.0.mlp.down_proj.weight_scale_inv": down_scale,
        "model.layers.0.linear_attn.in_proj_qkv.weight": in_qkv_q,
        "model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv": in_qkv_scale,
        "model.layers.0.linear_attn.in_proj_b.weight": in_b_bf16,
        "lm_head.weight": lm_head_bf16,
    }
    safetensors_torch.save_file(tensors, str(tmp_path / shard_name))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, shard_name)}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 4,
                "vocab_size": 8,
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )

    class _FakeQwen35NativeBridge:
        def __init__(self) -> None:
            self.dtype = torch.bfloat16
            self.hf_config = SimpleNamespace(
                model_type="qwen3_5",
                quantization_config={
                    "quant_method": "fp8",
                    "weight_block_size": [2, 2],
                },
            )

        def _get_actual_hf_path(self, path: str) -> str:
            return path

        def _get_safetensor_io(self, path: str) -> SafeTensorIO:
            return SafeTensorIO(path)

        def load_weights(self, models, path: str) -> None:
            io = self._get_safetensor_io(path)
            loaded = io.load_some_hf_weight(
                [
                    "model.layers.0.mlp.down_proj.weight",
                    "model.layers.0.linear_attn.in_proj_qkv.weight",
                    "model.layers.0.linear_attn.in_proj_b.weight",
                    "lm_head.weight",
                ]
            )
            model = models[0]
            assert loaded["lm_head.weight"].dtype is torch.bfloat16
            assert model.output_layer.weight.shape[0] == 1
            model.down_proj.data.copy_(loaded["model.layers.0.mlp.down_proj.weight"])
            model.in_proj.data.copy_(
                torch.cat(
                    [
                        loaded["model.layers.0.linear_attn.in_proj_qkv.weight"],
                        loaded["model.layers.0.linear_attn.in_proj_b.weight"],
                    ],
                    dim=0,
                )
            )

    model = _FakeGPTModel(hidden_size=4, vocab_size=8, post_process=True)
    registry._replace_output_layer_with_value_head(
        model,
        SimpleNamespace(
            hidden_size=4, params_dtype=torch.bfloat16, sequence_parallel=False
        ),
    )
    model.down_proj = torch.nn.Parameter(torch.empty(2, 4, dtype=torch.bfloat16))
    model.in_proj = torch.nn.Parameter(torch.empty(3, 4, dtype=torch.bfloat16))

    bridge = _FakeQwen35NativeBridge()
    orig_get_io = bridge._get_safetensor_io

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.bridge_cls = "mbridge"
    engine.bridge = bridge
    engine.model = [model]
    engine.dtype = torch.bfloat16
    engine.fp8_direct_convert = False
    engine.hf_config = bridge.hf_config
    engine.quantization_config = bridge.hf_config.quantization_config
    engine.config = SimpleNamespace(is_critic=True)

    engine._load_model_from_hf(str(tmp_path))

    assert bridge._get_safetensor_io == orig_get_io
    assert "_get_safetensor_io" not in bridge.__dict__
    expected_down = torch.tensor(
        [[0.04, 0.04, 2.0, 2.0], [0.04, 0.04, 2.0, 2.0]],
        dtype=torch.bfloat16,
    )
    expected_in = torch.tensor(
        [
            [0.2, 0.2, 0.5, 0.5],
            [0.2, 0.2, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(model.down_proj, expected_down, rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(model.in_proj, expected_in, rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(
        model.output_layer.weight,
        torch.zeros(1, 4, dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )


def test_dequant_fp8_safetensor_io_3d_experts_one_weight_and_missing_scale(
    tmp_path,
) -> None:
    MegatronEngine, hf_load, _ = _load_mcore_test_modules()
    safetensors_torch = pytest.importorskip("safetensors.torch")
    from mbridge.core.safetensor_io import SafeTensorIO

    shard_name = "model-00001-of-00001.safetensors"
    experts_q = torch.tensor(
        [
            [
                [2.0, 2.0, 4.0, 4.0],
                [2.0, 2.0, 4.0, 4.0],
                [2.0, 2.0, 4.0, 4.0],
            ],
            [
                [8.0, 8.0, 12.0, 12.0],
                [8.0, 8.0, 12.0, 12.0],
                [8.0, 8.0, 12.0, 12.0],
            ],
        ],
        dtype=torch.float32,
    ).to(torch.float8_e4m3fn)
    experts_scale = torch.tensor(
        [
            [[0.5, 0.25], [0.1, 0.05]],
            [[0.25, 0.5], [0.05, 0.1]],
        ],
        dtype=torch.float32,
    )
    unscaled_q = torch.ones((2, 2), dtype=torch.float32).to(torch.float8_e4m3fn)

    tensors = {
        "model.layers.0.mlp.experts.down_proj": experts_q,
        "model.layers.0.mlp.experts.down_proj_scale_inv": experts_scale,
        "model.layers.0.mlp.gate_proj.weight": unscaled_q,
    }
    safetensors_torch.save_file(tensors, str(tmp_path / shard_name))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(tensors, shard_name)}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "hidden_size": 4, "vocab_size": 8}),
        encoding="utf-8",
    )

    def orig_get_io(path: str) -> SafeTensorIO:
        return SafeTensorIO(path)

    bridge = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_config=SimpleNamespace(quantization_config={"weight_block_size": [2, 2]}),
        _get_safetensor_io=orig_get_io,
    )

    with hf_load.dequant_fp8_safetensor_io(bridge):
        io = bridge._get_safetensor_io(str(tmp_path))
        loaded_one = io.load_one_hf_weight("model.layers.0.mlp.experts.down_proj")
        loaded_many = io.load_some_hf_weight(["model.layers.0.mlp.experts.down_proj"])
        with pytest.raises(RuntimeError, match="no matching scale tensor"):
            io.load_one_hf_weight("model.layers.0.mlp.gate_proj.weight")

    assert bridge._get_safetensor_io is orig_get_io

    expected = torch.tensor(
        [
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [0.2, 0.2, 0.2, 0.2],
            ],
            [
                [2.0, 2.0, 6.0, 6.0],
                [2.0, 2.0, 6.0, 6.0],
                [0.4, 0.4, 1.2, 1.2],
            ],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(loaded_one, expected, rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(
        loaded_many["model.layers.0.mlp.experts.down_proj"],
        expected,
        rtol=1e-2,
        atol=1e-4,
    )

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.bridge_cls = "mbridge"
    engine.model = [object()]
    engine.fp8_direct_convert = True
    engine.hf_config = SimpleNamespace(model_type="qwen3_5")
    engine.config = SimpleNamespace(is_critic=False)
    with pytest.raises(NotImplementedError, match="fp8_direct_convert=True"):
        engine._load_model_from_hf(str(tmp_path))
