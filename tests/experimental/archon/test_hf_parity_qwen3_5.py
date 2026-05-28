from __future__ import annotations

import gc
from typing import Any

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

pytest.importorskip("transformers", minversion="5.2.0")

from tests.experimental.archon.utils import DENSE_MODEL_PATHS, setup_environment

from areal.experimental.models.archon import get_model_spec, is_supported_model
from areal.infra.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

MODEL_KEY = "qwen3_5"
PARTIAL_LAYERS = 4
SEQ_LEN = 32


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    n = min(a_f.numel(), b_f.numel())
    if n == 0:
        return 1.0
    a_f = a_f[:n]
    b_f = b_f[:n]
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()


def _capture_hf(
    hf_model: Any, n_layers: int
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    out: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _mk(name: str):
        def _hook(module: torch.nn.Module, args: tuple[Any, ...], value: Any) -> None:
            if isinstance(value, torch.Tensor):
                out[name] = value.detach().clone()
            elif (
                isinstance(value, tuple)
                and len(value) > 0
                and isinstance(value[0], torch.Tensor)
            ):
                out[name] = value[0].detach().clone()

        return _hook

    handles.append(hf_model.model.embed_tokens.register_forward_hook(_mk("emb")))
    for i in range(n_layers):
        layer = hf_model.model.layers[i]
        handles.append(
            layer.input_layernorm.register_forward_hook(_mk(f"layer{i}_attn_norm"))
        )
        # Hybrid: full_attention has self_attn, linear_attention has linear_attn
        if hasattr(layer, "self_attn") and layer.self_attn is not None:
            handles.append(layer.self_attn.register_forward_hook(_mk(f"layer{i}_attn")))
        if hasattr(layer, "linear_attn") and layer.linear_attn is not None:
            handles.append(
                layer.linear_attn.register_forward_hook(_mk(f"layer{i}_attn"))
            )
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(
                _mk(f"layer{i}_ffn_norm")
            )
        )
        handles.append(layer.mlp.register_forward_hook(_mk(f"layer{i}_ffn")))
        handles.append(layer.register_forward_hook(_mk(f"layer{i}_out")))
    handles.append(hf_model.model.norm.register_forward_hook(_mk("final_norm")))
    return out, handles


def _capture_archon(
    archon_model: Any, n_layers: int, layer_types: list[str] | None
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    out: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def _mk(name: str):
        def _hook(module: torch.nn.Module, args: tuple[Any, ...], value: Any) -> None:
            if isinstance(value, torch.Tensor):
                out[name] = value.detach().clone()
            elif (
                isinstance(value, tuple)
                and len(value) > 0
                and isinstance(value[0], torch.Tensor)
            ):
                out[name] = value[0].detach().clone()

        return _hook

    handles.append(archon_model.tok_embeddings.register_forward_hook(_mk("emb")))
    for i in range(n_layers):
        layer = archon_model.layers[str(i)]
        lt = layer_types[i] if layer_types is not None else "full_attention"
        handles.append(
            layer.attention_norm.register_forward_hook(_mk(f"layer{i}_attn_norm"))
        )
        if lt == "full_attention" and layer.attention is not None:
            handles.append(layer.attention.register_forward_hook(_mk(f"layer{i}_attn")))
        elif layer.linear_attn is not None:
            handles.append(
                layer.linear_attn.register_forward_hook(_mk(f"layer{i}_attn"))
            )
        handles.append(layer.ffn_norm.register_forward_hook(_mk(f"layer{i}_ffn_norm")))
        if layer.feed_forward is not None:
            handles.append(
                layer.feed_forward.register_forward_hook(_mk(f"layer{i}_ffn"))
            )
        handles.append(layer.register_forward_hook(_mk(f"layer{i}_out")))
    handles.append(archon_model.norm.register_forward_hook(_mk("final_norm")))
    return out, handles


@pytest.mark.slow
def test_hf_parity_qwen3_5() -> None:
    setup_environment()

    model_path = DENSE_MODEL_PATHS.get(MODEL_KEY)
    if model_path is None:
        pytest.skip(f"{MODEL_KEY} path not configured")

    dtype = torch.bfloat16
    device = torch.device(current_platform.device_type)

    full_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_full = (
        full_config.text_config if hasattr(full_config, "text_config") else full_config
    )

    partial_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if hasattr(partial_config, "text_config"):
        partial_config.text_config.num_hidden_layers = min(
            PARTIAL_LAYERS, int(partial_config.text_config.num_hidden_layers)
        )
        text_cfg = partial_config.text_config
    else:
        partial_config.num_hidden_layers = min(
            PARTIAL_LAYERS, int(partial_config.num_hidden_layers)
        )
        text_cfg = partial_config

    n_layers = int(text_cfg.num_hidden_layers)

    torch.manual_seed(42)
    input_ids = torch.randint(
        100, int(text_full.vocab_size) - 100, (1, SEQ_LEN), device=device
    )
    cu_seqlens = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=device)

    hf_model_any: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=partial_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    hf_model = hf_model_any.to(device).eval()

    hf_out, hf_handles = _capture_hf(hf_model, n_layers)
    hf_state_dict = {k: v.cpu() for k, v in hf_model.state_dict().items()}

    model_type = partial_config.model_type
    if not is_supported_model(model_type):
        pytest.skip(f"Model type {model_type} not supported")
    spec = get_model_spec(model_type)
    model_args = spec.model_args_class.from_hf_config(partial_config, is_critic=False)

    with torch.device(device):
        archon_model: Any = spec.model_class(model_args)

    adapter = spec.state_dict_adapter_class(partial_config)
    archon_sd = adapter.from_hf(hf_state_dict)
    load_ret = archon_model.load_state_dict(archon_sd, strict=False)
    assert len(load_ret.missing_keys) == 0
    assert len(load_ret.unexpected_keys) == 0

    archon_model = archon_model.to(dtype).eval()
    archon_model.init_buffers(device)

    layer_types = getattr(model_args, "layer_types", None)
    ar_out, ar_handles = _capture_archon(archon_model, n_layers, layer_types)

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
        archon_logits = archon_model(
            input_ids, positions=None, cu_seqlens=cu_seqlens, max_seqlen=SEQ_LEN
        )

    for h in hf_handles + ar_handles:
        h.remove()

    # --- Per-op diagnostics ---
    keys = [k for k in sorted(hf_out.keys()) if k in ar_out]
    print(f"\n[{MODEL_KEY} PER-OP]")
    trend: list[tuple[str, float]] = []
    for k in keys:
        hf_t = hf_out[k].float()
        ar_t = ar_out[k].float()
        if hf_t.shape != ar_t.shape:
            hf_t = hf_t.reshape(-1)
            ar_t = ar_t.reshape(-1)
            n = min(hf_t.numel(), ar_t.numel())
            hf_t = hf_t[:n]
            ar_t = ar_t[:n]
        diff = (hf_t - ar_t).abs()
        cos = _cos_sim(hf_t, ar_t)
        print(
            f"{k:24s} max={diff.max().item():.6f} mean={diff.mean().item():.6f} cos={cos:.6f}"
        )
        if k.startswith("layer") and k.endswith("_out"):
            trend.append((k, cos))

    if trend:
        print(f"\n[{MODEL_KEY} COS SIM TREND: layer*_out]")
        prev: float | None = None
        for name, cos in trend:
            if prev is None:
                print(f"{name:24s} cos={cos:.6f} delta=---")
            else:
                print(f"{name:24s} cos={cos:.6f} delta={cos - prev:+.6f}")
            prev = cos

    # --- Layer type info ---
    if layer_types is not None:
        layer_tags = [(i, layer_types[i]) for i in range(n_layers)]
        print(f"[{MODEL_KEY} LAYER TYPES] {layer_tags}")

    # --- Logits comparison ---
    logits_diff = (hf_logits.float() - archon_logits.float()).abs()
    max_diff = logits_diff.max().item()
    mean_diff = logits_diff.mean().item()
    logits_cos = _cos_sim(hf_logits, archon_logits)

    print(
        f"\n[{MODEL_KEY} LOGITS] max={max_diff:.6f} mean={mean_diff:.6f} "
        f"cos={logits_cos:.6f}"
    )

    assert max_diff < 6.0
    assert mean_diff < 0.5
    assert logits_cos > 0.95

    del hf_model, archon_model
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()
