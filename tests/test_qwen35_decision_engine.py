# SPDX-License-Identifier: Apache-2.0
"""CPU tests for MegatronEngine PointerHead wiring."""

from types import SimpleNamespace

import pytest
import torch

from tests.test_megatron_bridge_config import _load_function

pytest.importorskip("megatron.core")

from areal.models.mcore.qwen35_decision import DecisionHeadMeta

ENGINE_SOURCE = "areal/engine/megatron_engine.py"


def _method(name, extra=None):
    namespace = {
        "torch": torch,
        "Any": object,
        "is_qwen3_5_model": lambda model_type: model_type
        in {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"},
        "_QWEN35_LORA_TARGET_PRESETS": {
            "all": [
                "linear_qkv",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
                "in_proj",
                "out_proj",
            ],
            "dense": ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            "attn": ["linear_qkv", "linear_proj", "in_proj", "out_proj"],
            "qv": ["linear_qkv"],
        },
        "DecisionHeadMeta": DecisionHeadMeta,
        "PointerHead": type("PointerHead", (), {}),
        "load_decision_pointer_head": lambda *a, **k: None,
        "replace_output_layer_with_pointer_head": lambda *a, **k: None,
        "save_decision_pointer_head": lambda *a, **k: None,
        "unwrap_to_gpt_model": lambda chunk: chunk,
        "reassemble_cp_packed_logprobs": lambda output, *_: output,
        "unpad_logits": lambda output, *a, **k: output,
    }
    if extra:
        namespace.update(extra)
    return _load_function(ENGINE_SOURCE, name, namespace)


def test_qwen35_lora_presets_include_gdn_only_for_all_and_attn():
    resolve = _method("_bridge_lora_target_modules")
    engine = SimpleNamespace(
        config=SimpleNamespace(target_modules=[], lora_targets_preset="all"),
        hf_config=SimpleNamespace(model_type="qwen3_5"),
    )
    assert resolve(engine) == [
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",
        "in_proj",
        "out_proj",
    ]
    engine.config.lora_targets_preset = "dense"
    assert resolve(engine) == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    engine.config.lora_targets_preset = "attn"
    assert resolve(engine) == ["linear_qkv", "linear_proj", "in_proj", "out_proj"]
    engine.config.target_modules = ["linear_qkv", "output_layer", "q"]
    assert resolve(engine) == ["linear_qkv"]
    engine.config.target_modules = []
    engine.hf_config.model_type = "qwen3"
    engine.config.lora_targets_preset = "all"
    assert resolve(engine) == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]


def test_pre_wrap_attaches_head_before_lora_and_unfreezes():
    order = []

    class Head:
        def parameters(self):
            param = torch.nn.Parameter(torch.ones(2))
            param.requires_grad = False
            self.param = param
            return [param]

    chunk = SimpleNamespace(output_layer=Head())

    def attach(model):
        order.append("attach")
        return model

    def enable(model):
        order.append("unfreeze")
        for item in model:
            for param in item.output_layer.parameters():
                param.requires_grad = True

    def lora_hook(model):
        order.append("lora")
        for item in model:
            for param in item.output_layer.parameters():
                param.requires_grad = False
        return model

    create = _method("_create_pre_wrap_hook")
    engine = SimpleNamespace(
        lora_mode=True,
        bridge_cls="megatron-bridge",
        config=SimpleNamespace(is_decision_model=True),
        bridge_lora=SimpleNamespace(
            set_params_to_save=lambda model: order.append("save")
        ),
        _create_megatron_bridge_lora_hook=lambda: lora_hook,
        _attach_pointer_heads=attach,
        _enable_pointer_head_gradients=enable,
    )
    hook = create(engine)
    out = hook([chunk])
    assert out[0] is chunk
    assert order == ["attach", "lora", "unfreeze", "save"]
    assert chunk.output_layer.param.requires_grad is True


def test_decision_loss_skips_logprobs_and_scales():
    compute = _method("_compute_logprobs_and_loss")
    seen = {}

    def loss_fn(output, inputs):
        seen["output"] = output
        seen["inputs"] = inputs
        return output.sum()

    engine = SimpleNamespace(
        config=SimpleNamespace(is_decision_model=True, is_critic=False),
        enable_tree_training=False,
        _reassemble_and_unpad_output=lambda output, inputs: (
            output * 2,
            {k: v for k, v in inputs.items() if not k.startswith("_cp_")},
        ),
    )
    output = torch.ones(3, 4)
    inputs = {"token_roles": torch.tensor([2, 1, 0]), "_cp_padded_cu_seqlens": True}
    loss = compute(
        engine,
        output,
        inputs,
        loss_fn,
        lambda mb: torch.tensor(2.0),
        torch.tensor(4.0),
        loss_multiplier=3.0,
    )
    torch.testing.assert_close(seen["output"], output * 2)
    assert "_cp_padded_cu_seqlens" not in seen["inputs"]
    torch.testing.assert_close(loss, (output * 2).sum() * (2.0 / 4.0) * 3.0)


def test_decision_forward_returns_unpadded_projections():
    compute = _method("_compute_forward_result")
    output = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    engine = SimpleNamespace(
        config=SimpleNamespace(is_decision_model=True, is_critic=False),
        enable_tree_training=False,
        _reassemble_and_unpad_output=lambda tensor, inputs: (tensor + 1, inputs),
    )
    result = compute(engine, output, {"input_ids": torch.arange(2)})
    torch.testing.assert_close(result, output + 1)


def test_decision_head_meta_marks_lora_and_full():
    meta_fn = _method("_decision_head_checkpoint_meta")
    engine = SimpleNamespace(
        config=SimpleNamespace(
            path="Qwen/Qwen3.5-4B",
            dtype="bfloat16",
            use_lora=True,
            lora_rank=16,
            decision_head_dim=256,
        )
    )
    meta = meta_fn(engine)
    assert meta.base == "Qwen/Qwen3.5-4B"
    assert meta.lora == 16
    assert meta.weights == "lora"
    assert meta.weights_dtype == "bf16"
    engine.config.use_lora = False
    engine.config.lora_rank = 32
    assert meta_fn(engine).weights == "full"
    assert meta_fn(engine).lora == 0


def test_load_hf_skips_output_layer_then_syncs_head():
    calls = {}

    class Bridge:
        def load_hf_weights(self, model, **kwargs):
            calls["load"] = kwargs

    load = _method(
        "_load_model_from_hf",
        extra={"load_weights_from_hf_with_mbridge_fast": lambda **k: None},
    )
    engine = SimpleNamespace(
        model=["chunk"],
        bridge_cls="megatron-bridge",
        config=SimpleNamespace(is_decision_model=True, is_critic=False),
        mcore_config=SimpleNamespace(enable_mtp_training=False),
        _mtp_head_dropped=False,
        bridge=Bridge(),
        _sync_decision_pointer_head=lambda path: calls.setdefault("sync", path),
    )
    load(engine, "/ckpt")
    assert calls["load"]["hf_path"] == "/ckpt"
    assert calls["load"]["allowed_mismatched_params"] == [
        "*output_layer*",
        "lm_head.weight",
    ]
    assert calls["sync"] == "/ckpt"
    assert engine._frozen_mtp_state is None


def test_save_hf_writes_head_and_relaxes_lm_head_export(tmp_path):
    calls = {}

    class Bridge:
        def save_hf_pretrained(self, model, path, source_path=None, strict=True):
            calls["strict"] = strict
            calls["path"] = path

        def save_hf_adapter(self, *args, **kwargs):
            raise AssertionError("LoRA adapter path should not run")

    save = _method(
        "_save_model_to_hf",
        extra={
            "os": __import__("os"),
            "dist": SimpleNamespace(
                get_rank=lambda: 0,
                barrier=lambda group=None: None,
            ),
            "save_decision_pointer_head": lambda model, path, meta=None: calls.update(
                head=path, meta=meta
            ),
            "save_critic_value_head": lambda *a, **k: None,
            "save_weights_to_hf_with_mbridge_fast": lambda **k: None,
            "patch_torch_all_gather_object": lambda: None,
            "current_platform": SimpleNamespace(synchronize=lambda: None),
        },
    )
    ckpt_dir = str(tmp_path / "decision-ckpt")
    engine = SimpleNamespace(
        model=["chunk"],
        bridge_cls="megatron-bridge",
        config=SimpleNamespace(
            is_decision_model=True,
            is_critic=False,
            use_lora=False,
            path="Qwen/Qwen3.5-4B",
            dtype="bfloat16",
            lora_rank=16,
            decision_head_dim=8,
        ),
        bridge=Bridge(),
        _mtp_head_dropped=False,
        _run_hf_rank_zero=lambda fn: fn(),
        _rebuild_index_from_saved_shards=lambda path: calls.setdefault("index", path),
        _decision_head_checkpoint_meta=_method(
            "_decision_head_checkpoint_meta"
        ).__get__(None, object),
        cpu_group=None,
    )
    engine._decision_head_checkpoint_meta = lambda: _method(
        "_decision_head_checkpoint_meta"
    )(engine)
    save(engine, ckpt_dir)
    assert calls["strict"] is False
    assert calls["index"] == ckpt_dir
    assert calls["head"] == ckpt_dir
    assert calls["meta"].weights == "full"
    assert calls["meta"].head_dim == 8


def test_rebuild_index_prunes_ghost_lm_head_before_frozen_mtp_export(tmp_path):
    import json
    from pathlib import Path

    from safetensors.torch import save_file

    from areal.engine.megatron_utils.frozen_mtp import FrozenMTPState

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    save_file(
        {
            "model.embed_tokens.weight": torch.ones(4, 4),
            "lm_head.weight": torch.ones(4, 4),
            "mtp.layers.0.weight": torch.full((2, 2), 3.0),
        },
        src / "model-00001-of-00001.safetensors",
        metadata={"format": "pt"},
    )
    source_index = {
        "metadata": {"total_size": 144},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00001.safetensors",
            "lm_head.weight": "model-00001-of-00001.safetensors",
            "mtp.layers.0.weight": "model-00001-of-00001.safetensors",
        },
    }
    (src / "model.safetensors.index.json").write_text(json.dumps(source_index))
    (src / "config.json").write_text(
        json.dumps({"text_config": {"mtp_num_hidden_layers": 1}})
    )
    frozen = FrozenMTPState.load(str(src))

    # Bridge strict=False export omits both replaced lm_head.weight and dropped mtp.*,
    # while still writing the source weight_map.
    save_file(
        {"model.embed_tokens.weight": torch.ones(4, 4)},
        dst / "model-00001-of-00001.safetensors",
        metadata={"format": "pt"},
    )
    (dst / "model.safetensors.index.json").write_text(json.dumps(source_index))
    (dst / "config.json").write_text(
        json.dumps({"text_config": {"mtp_num_hidden_layers": 0}})
    )

    rebuild = _method(
        "_rebuild_index_from_saved_shards",
        extra={"Path": Path, "json": json},
    )
    engine = SimpleNamespace(logger=SimpleNamespace(info=lambda *a, **k: None))
    rebuild(engine, str(dst))
    frozen.export(str(dst))

    exported_index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert set(exported_index["weight_map"]) == {
        "model.embed_tokens.weight",
        "mtp.layers.0.weight",
    }


def test_pointer_head_skips_fp32_lm_head_patch_and_rejects_lm_head_flags():
    from areal.api.cli_args import MegatronEngineConfig, TrainEngineConfig
    from areal.models.mcore.qwen35_decision import PointerHead as RealPointerHead
    from areal.models.mcore.registry import _enable_fp32_lm_head_forward

    class FakeGPT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_layer = RealPointerHead(8, dp=4)

    gpt = FakeGPT()
    assert _enable_fp32_lm_head_forward([gpt], enabled=True) == 0
    assert not hasattr(gpt.output_layer, "_areal_fp32_lm_head_enabled")

    from areal.engine.megatron_engine import MegatronEngine

    with pytest.raises(ValueError, match="enable_chunked_logits=False"):
        MegatronEngine(
            TrainEngineConfig(
                experiment_name="t",
                trial_name="t",
                path="Qwen/Qwen3.5-4B",
                is_decision_model=True,
                megatron=MegatronEngineConfig(
                    bridge_type="megatron-bridge",
                    enable_chunked_logits=True,
                ),
            )
        )
    with pytest.raises(ValueError, match="enable_mtp_training=True"):
        MegatronEngine(
            TrainEngineConfig(
                experiment_name="t",
                trial_name="t",
                path="Qwen/Qwen3.5-4B",
                is_decision_model=True,
                megatron=MegatronEngineConfig(
                    bridge_type="megatron-bridge",
                    enable_mtp_training=True,
                ),
            )
        )
