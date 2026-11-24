# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import MegatronEngineConfig


def _load_registry_module():
    pytest.importorskip("megatron.core")
    pytest.importorskip("mbridge")
    from areal.models.mcore import registry

    return registry


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
    registry = _load_registry_module()
    tf_config = SimpleNamespace(sequence_parallel=True)

    head = registry.ValueHead(input_size=6, output_size=1, config=tf_config, bias=True)

    assert head.weight.shape == (1, 6)
    assert getattr(head.weight, "sequence_parallel", False) is True
    torch.testing.assert_close(head.weight, torch.zeros(1, 6), rtol=0, atol=0)
    torch.testing.assert_close(head.bias, torch.zeros(1), rtol=0, atol=0)


def test_mbridge_critic_installs_value_head_before_ddp_and_updates_weights(
    monkeypatch,
) -> None:
    registry = _load_registry_module()
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
    registry = _load_registry_module()
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
    registry = _load_registry_module()
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
    registry = _load_registry_module()
    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(registry, "DDP", _FakeDDP)

    tf_config = SimpleNamespace(
        hidden_size=4, params_dtype=torch.float32, sequence_parallel=False
    )
    ddp_model = _FakeDDP(
        _FakeGPTModel(hidden_size=4, vocab_size=8, post_process=True),
    )
    registry._replace_output_layer_with_value_head(ddp_model.module, tf_config)

    with pytest.raises(RuntimeError, match="not registered in Megatron DDP"):
        registry._verify_critic_value_heads([ddp_model])


def test_critic_rejects_enable_mtp_and_drops_disabled_checkpoint_mtp(
    monkeypatch,
) -> None:
    registry = _load_registry_module()
    monkeypatch.setattr(registry, "GPTModel", _FakeGPTModel)
    monkeypatch.setattr(registry, "DDP", _FakeDDP)

    with pytest.raises(ValueError, match="do not support MTP"):
        registry.make_mcore_model(
            hf_config=SimpleNamespace(),
            tf_config=SimpleNamespace(
                hidden_size=4, params_dtype=torch.float32, sequence_parallel=False
            ),
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
