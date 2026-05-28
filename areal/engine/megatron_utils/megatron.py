# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import re

import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
from torch import Tensor
from torch.nn.parameter import Parameter

from areal.engine.megatron_utils.fp8 import (
    FP8BlockwiseTensorHelper,
    convert_fp8_helper_to_pytorch_fp8,
    get_block_size_from_config,
    quantize_params,
)
from areal.engine.megatron_utils.megatron_lora import (
    convert_qwen3_lora_to_hf,
    convert_qwen3_moe_lora_to_hf,
)


@functools.cache
def _accepts_hf_config(fn) -> bool:
    return "hf_config" in inspect.signature(fn).parameters


def _all_gather_and_concat(
    tensor: torch.Tensor,
    tp_size: int,
    tp_group,
    partition_dim: int,
    partition_stride: int,
    name: str,
    gated_linear_unit: bool = False,
) -> torch.Tensor:
    """All-gather tensor partitions and concatenate along partition dimension.

    When partition_stride > 1 (e.g., GLU/SwiGLU layers where gate and up projections
    are interleaved), each TP rank stores interleaved sub-blocks. After all-gather,
    these must be de-interleaved before concatenation to reconstruct the correct
    full tensor.
    """
    # mcore sets partition_stride=1 for ``linear_fc1.weight|bias`` even when
    # gated_linear_unit=True, but the per-rank storage is ``[gate_local | up_local]``
    # (what makes the in-place ``chunk(2, dim=-1)`` in MLP.forward valid). Without
    # de-interleaving here, plain cat yields ``[gate_r0 | up_r0 | gate_r1 | up_r1]``
    # and downstream ``chunk(2)`` mislabels mixed halves as gate_proj/up_proj.
    # Override to stride=2 to trigger the de-interleave below.
    # Covers both ``mlp.linear_fc1`` (language/vision) and
    # ``projection.encoder.linear_fc1`` (vision→language merger).
    if (
        gated_linear_unit
        and partition_stride == 1
        and ("linear_fc1.weight" in name or "linear_fc1.bias" in name)
    ):
        partition_stride = 2

    partitions = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(partitions, tensor, group=tp_group)

    # De-interleave strided partitions. With stride S, each rank stores S interleaved
    # sub-blocks. We split each partition into S chunks and regroup by stride index
    # so that all sub-blocks for stride 0 come first, then stride 1, etc.
    # Example (stride=2, tp=2): rank0=[gate_0|up_0], rank1=[gate_1|up_1]
    #   → [gate_0, gate_1, up_0, up_1] → cat → [gate_full | up_full]
    if partition_stride > 1:
        chunks = [p.chunk(partition_stride, dim=partition_dim) for p in partitions]
        partitions = [
            chunks[rank][s] for s in range(partition_stride) for rank in range(tp_size)
        ]

    # this is bug in megatron's grouped moe.
    partition_dim = (
        1 if "linear_fc2.weight" in name and partition_dim == 0 else partition_dim
    )

    return torch.cat(partitions, dim=partition_dim)


def _all_gather_fp8_tensor_and_concat(
    tensor,
    tp_size: int,
    tp_group,
    partition_dim: int,
    partition_stride: int,
    name: str,
    block_size: int = 128,
    gated_linear_unit: bool = False,
) -> FP8BlockwiseTensorHelper:
    """All-gather a Float8BlockwiseQTensor along the partition dimension.

    Returns FP8BlockwiseTensorHelper that wraps rowwise_data and rowwise_scale_inv.
    This allows conversion functions to work with FP8 tensors as regular tensors.
    """
    gathered_rowwise_data = _all_gather_and_concat(
        tensor._rowwise_data,
        tp_size,
        tp_group,
        partition_dim,
        partition_stride,
        name,
        gated_linear_unit=gated_linear_unit,
    )
    gathered_rowwise_scale_inv = _all_gather_and_concat(
        tensor._rowwise_scale_inv,
        tp_size,
        tp_group,
        partition_dim,
        partition_stride,
        name,
        gated_linear_unit=gated_linear_unit,
    )

    return FP8BlockwiseTensorHelper(
        gathered_rowwise_data, gathered_rowwise_scale_inv, block_size
    )


# Adapted from slime
def all_gather_param(
    name: str,
    param: Parameter | Tensor,
    fp8_direct_convert: bool = False,
    quantization_config: dict[str, int | str | list[str]] | None = None,
    duplicated_param_names: set[str] | None = None,
    gated_linear_unit: bool = False,
) -> torch.Tensor | FP8BlockwiseTensorHelper:
    if "expert_bias" in name:
        return param

    param_is_fp8 = is_float8tensor(param)

    if not hasattr(param, "tensor_model_parallel"):
        if param_is_fp8 and fp8_direct_convert:
            return param
        return param.data

    # Check if this param is truly NOT TP-sharded.
    # NOTE: TE unconditionally sets tensor_model_parallel=True on all Linear
    # weights, even for modules with parallel_mode='duplicated'. The original
    # getattr(param, "parallel_mode", ...) check was dead code because
    # parallel_mode is a module attribute, not a tensor attribute.
    # Use the caller-provided duplicated_param_names set for reliable detection.
    is_duplicated = (
        duplicated_param_names is not None and name in duplicated_param_names
    )
    if not param.tensor_model_parallel or is_duplicated:
        # NOTE: For FP8 tensors with direct conversion, return the tensor directly
        # without accessing .data to avoid dequantization (accessing .data on
        # QuantizedTensor triggers __torch_dispatch__ which dequantizes to bfloat16).
        # Otherwise, .data will implicitly convert TE FP8 to bf16, which will be
        # converted to PyTorch FP8 later in convert_to_hf.
        if param_is_fp8 and fp8_direct_convert:
            return param
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    partition_dim = param.partition_dim
    partition_stride = param.partition_stride

    # Handle FP8 tensors specially
    if param_is_fp8 and fp8_direct_convert:
        block_size = get_block_size_from_config(quantization_config)
        return _all_gather_fp8_tensor_and_concat(
            param,
            tp_size,
            tp_group,
            partition_dim,
            partition_stride,
            name,
            block_size,
            gated_linear_unit=gated_linear_unit,
        )

    # bf16/fp32
    param = _all_gather_and_concat(
        param.data,
        tp_size,
        tp_group,
        partition_dim,
        partition_stride,
        name,
        gated_linear_unit=gated_linear_unit,
    )
    return param


# Adapted from slime
def remove_padding(
    name: str, param: Parameter | Tensor | FP8BlockwiseTensorHelper, vocab_size: int
):
    if name in (
        "module.module.embedding.word_embeddings.weight",
        "module.module.output_layer.weight",
        "module.module.language_model.embedding.word_embeddings.weight",
        "module.module.language_model.output_layer.weight",
    ):
        return param[:vocab_size]
    return param


# Adapted from slime
def convert_qwen3moe_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = (
            tf_config.kv_channels
            if tf_config.kv_channels is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
    except (AttributeError, TypeError):
        head_dim = tf_config.hidden_size // tf_config.num_attention_heads
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups

    if tf_config.num_query_groups is None:
        raise ValueError("Qwen3-MoE models should have num_query_groups")

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                        up_weight,
                    ),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                        param,
                    ),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
                        up_weight,
                    ),
                ]
            elif rest == "linear_fc2.weight":
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
                        param,
                    )
                ]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(
                tf_config.num_query_groups, -1, head_dim, tf_config.hidden_size
            )
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, tf_config.hidden_size)
            k_param = k_param.reshape(-1, tf_config.hidden_size)
            v_param = v_param.reshape(-1, tf_config.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(tf_config.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    value_num_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "pre_mlp_layernorm.weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [
                (f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)
            ]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


def _vision_qkv_mcore_to_hf(param: Tensor, vision_num_heads: int) -> Tensor:
    """Convert vision encoder QKV from mcore interleaved to HF grouped format.

    mcore: per-head interleaved [num_heads, 3, head_dim, H] flattened to [3*H_v, H_v]
    HF:    grouped [3, num_heads, head_dim, H] flattened to [3*H_v, H_v]

    Reverse of mbridge Qwen2_5VLBridge._weight_to_mcore_format for vision QKV.

    Args:
        param: Vision QKV weight [3*hidden_vision, hidden_vision] or bias [3*hidden_vision]
        vision_num_heads: Number of attention heads in the vision encoder
    """
    hidden_vision = param.shape[0] // 3
    head_dim = hidden_vision // vision_num_heads
    # Vision encoders for both Qwen2.5-VL and Qwen3-VL set
    # num_query_groups == num_attention_heads (no GQA on the vision side); a
    # future VLM that breaks this would silently miscompile vision QKV here.
    assert head_dim * vision_num_heads * 3 == param.shape[0], (
        f"_vision_qkv_mcore_to_hf assumes vision has no GQA "
        f"(num_kv_heads == num_heads). Got param.shape[0]={param.shape[0]}, "
        f"vision_num_heads={vision_num_heads}, derived head_dim={head_dim}."
    )
    is_bias = param.ndim == 1

    if is_bias:
        x = param.view(vision_num_heads, 3, head_dim)
        return x.permute(1, 0, 2).contiguous().view(-1)
    else:
        in_features = param.shape[-1]
        x = param.view(vision_num_heads, 3, head_dim, in_features)
        return x.permute(1, 0, 2, 3).contiguous().view(-1, in_features)


def convert_qwen2_5_vl_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
    hf_config=None,
):
    """Convert Qwen2.5-VL Megatron parameters to HuggingFace format.

    Handles both vision_model and language_model parameters.
    Language model params are delegated to convert_qwen2_to_hf after
    stripping the 'language_model.' prefix.
    """
    # --- Language model: strip prefix and delegate ---
    _LM_PREFIX = "module.module.language_model."
    if name.startswith(_LM_PREFIX):
        lm_name = "module.module." + name[len(_LM_PREFIX) :]
        return convert_qwen2_to_hf(tf_config, lm_name, param)

    # --- megatron-bridge: vision tower stored in HF format under self.visual.* ---
    # (vs mbridge's self.vision_model.* in mcore format). Just strip the
    # "module.module." prefix and emit the HF name directly.
    _MB_VISUAL_PREFIX = "module.module.visual."
    if name.startswith(_MB_VISUAL_PREFIX):
        return [(name[len("module.module.") :], param)]

    # --- mbridge vision tower (mcore format) — direct mappings ---
    _VISION_DIRECT = {
        "module.module.vision_model.patch_embed.proj.weight": "visual.patch_embed.proj.weight",
        "module.module.vision_model.decoder.final_layernorm.weight": "visual.merger.ln_q.weight",
        "module.module.vision_model.projection.encoder.linear_fc1.weight": "visual.merger.mlp.0.weight",
        "module.module.vision_model.projection.encoder.linear_fc1.bias": "visual.merger.mlp.0.bias",
        "module.module.vision_model.projection.encoder.linear_fc2.weight": "visual.merger.mlp.2.weight",
        "module.module.vision_model.projection.encoder.linear_fc2.bias": "visual.merger.mlp.2.bias",
    }
    if name in _VISION_DIRECT:
        return [(_VISION_DIRECT[name], param)]

    # --- Vision model per-layer params ---
    vision_layers_pattern = (
        r"module\.module\.vision_model\.decoder\.layers\.(\d+)\.(.+)"
    )
    match = re.match(vision_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # Attention — vision QKV needs reordering from mcore interleaved
        # [num_heads, 3*head_dim, ...] to HF grouped [3*num_heads*head_dim, ...]
        if rest in (
            "self_attention.linear_qkv.weight",
            "self_attention.linear_qkv.bias",
        ):
            vision_num_heads = getattr(
                getattr(hf_config, "vision_config", None), "num_heads", None
            )
            if vision_num_heads is None:
                raise ValueError(
                    "hf_config.vision_config.num_heads is required for vision QKV "
                    "conversion. Pass hf_config to convert_to_hf()."
                )
            param = _vision_qkv_mcore_to_hf(param, vision_num_heads)
            hf_key = "weight" if "weight" in rest else "bias"
            return [(f"visual.blocks.{layer_idx}.attn.qkv.{hf_key}", param)]
        elif rest == "self_attention.linear_proj.weight":
            return [(f"visual.blocks.{layer_idx}.attn.proj.weight", param)]
        elif rest == "self_attention.linear_proj.bias":
            return [(f"visual.blocks.{layer_idx}.attn.proj.bias", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"visual.blocks.{layer_idx}.norm1.weight", param)]

        # MLP
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"visual.blocks.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"visual.blocks.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc1.bias":
            gate_bias, up_bias = param.chunk(2, dim=0)
            return [
                (f"visual.blocks.{layer_idx}.mlp.gate_proj.bias", gate_bias),
                (f"visual.blocks.{layer_idx}.mlp.up_proj.bias", up_bias),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"visual.blocks.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "mlp.linear_fc2.bias":
            return [(f"visual.blocks.{layer_idx}.mlp.down_proj.bias", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"visual.blocks.{layer_idx}.norm2.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


_QWEN3_VL_LM_PREFIX = "module.module.language_model."
_QWEN3_VL_LM_LAYER_RE = re.compile(
    r"module\.module\.language_model\.decoder\.layers\.(\d+)\.(.+)"
)
_QWEN3_VL_VISION_LAYER_RE = re.compile(
    r"module\.module\.vision_model\.decoder\.layers\.(\d+)\.(.+)"
)
_QWEN3_VL_VISION_DEEPSTACK_RE = re.compile(
    r"module\.module\.vision_model\.decoder\.deepstack_merger_list\.(\d+)\.(.+)"
)


def _convert_qwen3_vl_lm_global(name, param):
    """Top-level language-model tensors (embeddings, final norm, lm_head).

    Returns the converted name list if matched, else ``None``.
    """
    if name == "module.module.language_model.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.language_model.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]
    if name == "module.module.language_model.output_layer.weight":
        return [("lm_head.weight", param)]
    return None


def _convert_qwen3_vl_lm_attention(tf_config, layer_idx, rest, param):
    """Per-layer attention block shared by Qwen3-VL dense and MoE.

    Returns the converted name list if ``rest`` matches an attention key,
    else ``None`` so the caller can fall through to dense or MoE MLP logic.
    """
    if tf_config.num_query_groups is None:
        raise ValueError("Qwen3-VL text model requires num_query_groups")
    kv_channels = getattr(tf_config, "kv_channels", None)
    head_dim = (
        kv_channels
        if kv_channels is not None
        else tf_config.hidden_size // tf_config.num_attention_heads
    )
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups
    base = f"model.language_model.layers.{layer_idx}"

    if rest == "self_attention.linear_proj.weight":
        return [(f"{base}.self_attn.o_proj.weight", param)]
    if rest == "self_attention.linear_qkv.weight":
        p = param.view(tf_config.num_query_groups, -1, head_dim, tf_config.hidden_size)
        q, k, v = torch.split(
            p, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
        )
        return [
            (f"{base}.self_attn.q_proj.weight", q.reshape(-1, tf_config.hidden_size)),
            (f"{base}.self_attn.k_proj.weight", k.reshape(-1, tf_config.hidden_size)),
            (f"{base}.self_attn.v_proj.weight", v.reshape(-1, tf_config.hidden_size)),
        ]
    if rest == "self_attention.linear_qkv.bias":
        p = param.view(tf_config.num_query_groups, -1)
        q, k, v = torch.split(
            p,
            split_size_or_sections=[
                value_num_per_group * head_dim,
                head_dim,
                head_dim,
            ],
            dim=1,
        )
        return [
            (f"{base}.self_attn.q_proj.bias", q.contiguous().flatten()),
            (f"{base}.self_attn.k_proj.bias", k.contiguous().flatten()),
            (f"{base}.self_attn.v_proj.bias", v.contiguous().flatten()),
        ]
    if rest == "self_attention.linear_qkv.layer_norm_weight":
        return [(f"{base}.input_layernorm.weight", param)]
    if rest == "self_attention.q_layernorm.weight":
        return [(f"{base}.self_attn.q_norm.weight", param)]
    if rest == "self_attention.k_layernorm.weight":
        return [(f"{base}.self_attn.k_norm.weight", param)]
    return None


def _convert_qwen3_vl_vision_to_hf(name, param, hf_config):
    """Vision tower + deepstack merger conversion shared by dense and MoE.

    Mirrors mbridge.models.qwen3_vl.Qwen3VBaseBridge mappings.
    Raises ``ValueError`` if ``name`` does not match any vision-side tensor.
    """
    _VISION_DIRECT = {
        "module.module.vision_model.patch_embed.proj.weight": "model.visual.patch_embed.proj.weight",
        "module.module.vision_model.patch_embed.proj.bias": "model.visual.patch_embed.proj.bias",
        "module.module.vision_model.pos_embed.weight": "model.visual.pos_embed.weight",
        "module.module.vision_model.merger.patch_norm.weight": "model.visual.merger.norm.weight",
        "module.module.vision_model.merger.patch_norm.bias": "model.visual.merger.norm.bias",
        "module.module.vision_model.merger.linear_fc1.weight": "model.visual.merger.linear_fc1.weight",
        "module.module.vision_model.merger.linear_fc1.bias": "model.visual.merger.linear_fc1.bias",
        "module.module.vision_model.merger.linear_fc2.weight": "model.visual.merger.linear_fc2.weight",
        "module.module.vision_model.merger.linear_fc2.bias": "model.visual.merger.linear_fc2.bias",
    }
    if name in _VISION_DIRECT:
        return [(_VISION_DIRECT[name], param)]

    match = _QWEN3_VL_VISION_LAYER_RE.match(name)
    if match:
        layer_idx, rest = match.groups()
        base = f"model.visual.blocks.{layer_idx}"

        if rest in (
            "self_attention.linear_qkv.weight",
            "self_attention.linear_qkv.bias",
        ):
            vision_num_heads = getattr(
                getattr(hf_config, "vision_config", None), "num_heads", None
            )
            if vision_num_heads is None:
                raise ValueError(
                    "hf_config.vision_config.num_heads is required for vision QKV "
                    "conversion. Pass hf_config to convert_to_hf()."
                )
            param = _vision_qkv_mcore_to_hf(param, vision_num_heads)
            kind = "weight" if rest.endswith("weight") else "bias"
            return [(f"{base}.attn.qkv.{kind}", param)]
        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.attn.proj.weight", param)]
        if rest == "self_attention.linear_proj.bias":
            return [(f"{base}.attn.proj.bias", param)]
        if rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{base}.norm1.weight", param)]
        if rest == "self_attention.linear_qkv.layer_norm_bias":
            return [(f"{base}.norm1.bias", param)]
        # Qwen3-VL vision MLP is non-gated: 1:1 mapping (no chunk)
        if rest == "mlp.linear_fc1.weight":
            return [(f"{base}.mlp.linear_fc1.weight", param)]
        if rest == "mlp.linear_fc1.bias":
            return [(f"{base}.mlp.linear_fc1.bias", param)]
        if rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.linear_fc2.weight", param)]
        if rest == "mlp.linear_fc2.bias":
            return [(f"{base}.mlp.linear_fc2.bias", param)]
        if rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{base}.norm2.weight", param)]
        if rest == "mlp.linear_fc1.layer_norm_bias":
            return [(f"{base}.norm2.bias", param)]

    match = _QWEN3_VL_VISION_DEEPSTACK_RE.match(name)
    if match:
        idx, rest = match.groups()
        base = f"model.visual.deepstack_merger_list.{idx}"
        if rest == "patch_norm.weight":
            return [(f"{base}.norm.weight", param)]
        if rest == "patch_norm.bias":
            return [(f"{base}.norm.bias", param)]
        if rest in (
            "linear_fc1.weight",
            "linear_fc1.bias",
            "linear_fc2.weight",
            "linear_fc2.bias",
        ):
            return [(f"{base}.{rest}", param)]

    raise ValueError(f"Unknown Qwen3-VL parameter: {name}")


def convert_qwen3_vl_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
    hf_config=None,
):
    """Convert dense Qwen3-VL Megatron parameters to HuggingFace format.

    Qwen3-VL-MoE is handled by ``convert_qwen3_vl_moe_to_hf``; the registry
    in ``_CONVERSION_FN_REGISTRY`` MUST register ``qwen3_vl_moe`` BEFORE
    ``qwen3_vl`` (substring matching dispatches the first hit).

    HF naming differs from Qwen2.5-VL:
    - Vision: ``model.visual.*`` (extra ``model.`` prefix)
    - Language: ``model.language_model.layers.*`` (extra ``language_model``)
    - Text Q/K norms: ``self_attn.{q,k}_norm.weight`` (vs Qwen2's missing q/k norm)
    - Vision MLP is non-gated: ``linear_fc1`` / ``linear_fc2`` map 1:1 (no chunk).
    - Vision norms have bias; vision merger uses ``patch_norm`` and
      ``linear_fc{1,2}`` directly (no separate projection encoder).
    - New tensors: ``model.visual.pos_embed``, ``patch_embed.proj.bias``,
      ``deepstack_merger_list.{i}.{norm,linear_fc1,linear_fc2}``.

    The mapping mirrors mbridge.models.qwen3_vl.Qwen3VLBridge (the source of
    truth used by ``update_weights``).
    """

    converted = _convert_qwen3_vl_lm_global(name, param)
    if converted is not None:
        return converted

    if name.startswith(_QWEN3_VL_LM_PREFIX):
        match = _QWEN3_VL_LM_LAYER_RE.match(name)
        if match:
            layer_idx, rest = match.groups()
            attn = _convert_qwen3_vl_lm_attention(tf_config, layer_idx, rest, param)
            if attn is not None:
                return attn

            base = f"model.language_model.layers.{layer_idx}"
            if rest == "mlp.linear_fc1.weight":
                gate, up = param.chunk(2, dim=0)
                return [
                    (f"{base}.mlp.gate_proj.weight", gate),
                    (f"{base}.mlp.up_proj.weight", up),
                ]
            if rest == "mlp.linear_fc1.layer_norm_weight":
                return [(f"{base}.post_attention_layernorm.weight", param)]
            if rest == "mlp.linear_fc2.weight":
                return [(f"{base}.mlp.down_proj.weight", param)]

        raise ValueError(f"Unknown Qwen3-VL language-model parameter: {name}")

    return _convert_qwen3_vl_vision_to_hf(name, param, hf_config)


def convert_qwen3_vl_moe_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
    hf_config=None,
):
    """Convert Qwen3-VL-MoE Megatron parameters to HuggingFace format.

    Mirrors ``convert_qwen3moe_to_hf`` (per-expert flat HF keys, no transpose,
    stateless) so the XCCL ``update_weights`` path to vLLM/SGLang reuses the
    same shape contract that ships today for Qwen3-MoE. Vision tower + language
    attention + global LM tensors share the dense Qwen3-VL converter helpers.

    Differences from dense Qwen3-VL:
    - Per-layer dense MLP (``mlp.linear_fc{1,2}``) is replaced by per-expert
      ``mlp.experts.linear_fc{1,2}.weight{idx}``, plus router and pre-MLP layernorm.
    - No shared experts (verified against mbridge ``Qwen3VLMoEBridge._MLP_MAPPING``).
    - No router expert_bias (mbridge sets ``moe_router_load_balancing_type="none"``).

    Registry MUST place ``qwen3_vl_moe`` before ``qwen3_vl``, ``qwen3_moe``, and
    ``qwen3`` in ``_CONVERSION_FN_REGISTRY`` due to substring-match dispatch.
    """

    converted = _convert_qwen3_vl_lm_global(name, param)
    if converted is not None:
        return converted

    if name.startswith(_QWEN3_VL_LM_PREFIX):
        match = _QWEN3_VL_LM_LAYER_RE.match(name)
        if match:
            layer_idx, rest = match.groups()
            attn = _convert_qwen3_vl_lm_attention(tf_config, layer_idx, rest, param)
            if attn is not None:
                return attn

            base = f"model.language_model.layers.{layer_idx}"

            expert_match = re.match(r"mlp\.experts\.(.+)\.weight(\d+)", rest)
            if expert_match:
                fc_kind, expert_idx = expert_match.groups()
                if fc_kind == "linear_fc1":
                    gate, up = param.chunk(2, dim=0)
                    return [
                        (
                            f"{base}.mlp.experts.{expert_idx}.gate_proj.weight",
                            gate,
                        ),
                        (
                            f"{base}.mlp.experts.{expert_idx}.up_proj.weight",
                            up,
                        ),
                    ]
                if fc_kind == "linear_fc2":
                    return [
                        (
                            f"{base}.mlp.experts.{expert_idx}.down_proj.weight",
                            param,
                        )
                    ]
                raise ValueError(f"Unknown Qwen3-VL-MoE expert parameter: {name}")

            if rest == "mlp.router.weight":
                return [(f"{base}.mlp.gate.weight", param)]
            if rest == "pre_mlp_layernorm.weight":
                return [(f"{base}.post_attention_layernorm.weight", param)]

            # Dense MLP fallback for layers where ``decoder_sparse_step > 1``
            # leaves some layers non-sparse. Qwen3-VL-30B-A3B-Instruct uses
            # ``decoder_sparse_step=1`` so every layer is MoE and these branches
            # are dead, but the same converter is expected to cover future
            # variants with mixed dense/sparse layouts. Mirrors the dense-MLP
            # tail of ``convert_qwen3moe_to_hf``.
            if rest == "mlp.linear_fc1.weight":
                gate, up = param.chunk(2, dim=0)
                return [
                    (f"{base}.mlp.gate_proj.weight", gate),
                    (f"{base}.mlp.up_proj.weight", up),
                ]
            if rest == "mlp.linear_fc1.layer_norm_weight":
                return [(f"{base}.post_attention_layernorm.weight", param)]
            if rest == "mlp.linear_fc2.weight":
                return [(f"{base}.mlp.down_proj.weight", param)]

        raise ValueError(f"Unknown Qwen3-VL-MoE language-model parameter: {name}")

    return _convert_qwen3_vl_vision_to_hf(name, param, hf_config)


# Adapted from slime
def convert_qwen2_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = (
            tf_config.kv_channels
            if tf_config.kv_channels is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
    except (AttributeError, TypeError):
        head_dim = tf_config.hidden_size // tf_config.num_attention_heads
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups

    if tf_config.num_query_groups is None:
        raise ValueError("Qwen2 models should have num_query_groups")

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(
                tf_config.num_query_groups, -1, head_dim, tf_config.hidden_size
            )
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, tf_config.hidden_size)
            k_param = k_param.reshape(-1, tf_config.hidden_size)
            v_param = v_param.reshape(-1, tf_config.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(tf_config.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    value_num_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


# Adapted from slime
def convert_deepseekv3_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = (
            tf_config.kv_channels
            if tf_config.kv_channels is not None
            else tf_config.hidden_size // tf_config.num_attention_heads
        )
    except (AttributeError, TypeError):
        head_dim = tf_config.hidden_size // tf_config.num_attention_heads
    value_num_per_group = tf_config.num_attention_heads // tf_config.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                        up_weight,
                    ),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                        param,
                    ),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
                        up_weight,
                    ),
                ]
            elif rest == "linear_fc2.weight":
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
                        param,
                    )
                ]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_q_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_proj.weight", param)]
        elif rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_proj.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_b_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(tf_config.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[
                    value_num_per_group * head_dim,
                    head_dim,
                    head_dim,
                ],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif (
            rest == "self_attention.linear_qkv.layer_norm_weight"
            or rest == "input_layernorm.weight"
        ):
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "self_attention.linear_kv_down_proj.weight":
            return [
                (f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", param)
            ]
        elif rest == "self_attention.linear_kv_up_proj.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", param)
            ]
        elif rest == "self_attention.linear_kv_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [
                (f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)
            ]

    raise ValueError(f"Unknown parameter name: {name}")


# BailingMoeV2_5 weight conversion
#
# BailingMoe HF uses "attention." prefix (not "self_attn.").
# Lightning layers: fused QKV (query_key_value), gate (g_proj), gate norm (g_norm),
#                   output proj (dense), Q/K norms (query_layernorm/key_layernorm)
# MLA layers: separate Q/KV low-rank projections (q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj),
#             output proj (dense), Q/KV norms (q_a_layernorm, kv_a_layernorm)
def convert_bailingmoe_to_hf(
    tf_config: TransformerConfig,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
):
    """Convert BailingMoeV2_5 megatron-core weights to HuggingFace format.

    BailingMoeV2_5 has two attention types per layer:
    - Lightning Attention: uses fused QKV (linear_qkv) + gate projection (linear_gate)
    - MLA: uses separate Q/KV down/up projections

    The layer type is determined by the parameter name structure:
    - Lightning layers have: linear_qkv, linear_gate, gate_norm
    - MLA layers have: linear_q_down_proj, linear_q_up_proj, linear_kv_down_proj, etc.

    HF naming convention uses "attention." prefix (not "self_attn.").
    """
    if "_extra_state" in name:
        return []
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.word_embeddings.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # === MoE experts ===
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                        up_weight,
                    ),
                ]
            elif rest == "linear_fc2":
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                        param,
                    ),
                ]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # === Shared experts ===
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
                        up_weight,
                    ),
                ]
            elif rest == "linear_fc2.weight":
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight",
                        param,
                    )
                ]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        # === Dense MLP ===
        if rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

        # === MoE router ===
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.expert_bias", param)]

        # === Lightning Attention layers (fused QKV) ===
        elif rest == "self_attention.linear_qkv.weight":
            # Mcore stores QKV in interleaved [H, 3, D] format: [q0,k0,v0, q1,k1,v1,...]
            # HF stores QKV in concatenated [Q_all, K_all, V_all] format
            # Convert interleaved → concatenated for HF checkpoint
            num_heads = tf_config.num_attention_heads
            head_dim = param.shape[0] // (num_heads * 3)
            hidden = param.shape[1]
            qkv = param.view(num_heads, 3, head_dim, hidden)
            q = qkv[:, 0].reshape(-1, hidden)  # [H*D, hidden]
            k = qkv[:, 1].reshape(-1, hidden)
            v = qkv[:, 2].reshape(-1, hidden)
            param = torch.cat([q, k, v], dim=0)  # [3*H*D, hidden]
            return [
                (
                    f"model.layers.{layer_idx}.attention.query_key_value.weight",
                    param,
                )
            ]
        elif rest == "self_attention.linear_qkv.bias":
            return [
                (
                    f"model.layers.{layer_idx}.attention.query_key_value.bias",
                    param,
                )
            ]

        # === Lightning gate projection and norm ===
        elif rest == "self_attention.linear_gate.weight":
            return [(f"model.layers.{layer_idx}.attention.g_proj.weight", param)]
        elif rest == "self_attention.linear_gate.bias":
            return [(f"model.layers.{layer_idx}.attention.g_proj.bias", param)]
        elif rest == "self_attention.gate_norm.weight":
            return [(f"model.layers.{layer_idx}.attention.g_norm.weight", param)]

        # === MLA layers (separate Q/KV projections) ===
        elif rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.attention.q_a_proj.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.attention.q_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.attention.q_b_proj.weight", param)]
        elif rest == "self_attention.linear_kv_down_proj.weight":
            return [
                (
                    f"model.layers.{layer_idx}.attention.kv_a_proj_with_mqa.weight",
                    param,
                )
            ]
        elif rest == "self_attention.linear_kv_up_proj.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.attention.kv_a_layernorm.weight", param)
            ]
        elif rest == "self_attention.linear_kv_up_proj.weight":
            return [(f"model.layers.{layer_idx}.attention.kv_b_proj.weight", param)]
        elif rest == "self_attention.linear_q_proj.weight":
            return [(f"model.layers.{layer_idx}.attention.q_proj.weight", param)]

        # === Output projection (both layer types) -> attention.dense ===
        elif rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.attention.dense.weight", param)]

        # === LayerNorm weights ===
        elif (
            rest == "self_attention.linear_qkv.layer_norm_weight"
            or rest == "input_layernorm.weight"
        ):
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]
        elif rest == "pre_mlp_layernorm.weight":
            return [
                (f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)
            ]

        # === Lightning Q/K norms ===
        elif rest == "self_attention.q_layernorm.weight":
            return [
                (f"model.layers.{layer_idx}.attention.query_layernorm.weight", param)
            ]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.attention.key_layernorm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


# Adapted from slime
# A registry for conversion functions is more extensible.
# Ordering matters: ``convert_to_hf`` dispatches on the FIRST substring hit, so
# longer/more-specific keys MUST come before shorter ones that are substrings of
# them. In particular, ``qwen3_vl_moe`` must precede ``qwen3_vl``,
# ``qwen3_moe``, and ``qwen3``.
_CONVERSION_FN_REGISTRY = {
    "qwen3_lora": convert_qwen3_lora_to_hf,
    "qwen2_lora": convert_qwen3_lora_to_hf,
    "qwen3_moe_lora": convert_qwen3_moe_lora_to_hf,
    "qwen2_5_vl": convert_qwen2_5_vl_to_hf,
    "qwen3_vl_moe": convert_qwen3_vl_moe_to_hf,
    "qwen3_vl": convert_qwen3_vl_to_hf,
    "qwen3_moe": convert_qwen3moe_to_hf,
    "qwen2": convert_qwen2_to_hf,
    "qwen3": convert_qwen2_to_hf,
    "deepseekv3": convert_deepseekv3_to_hf,
    "bailing_moe_v2": convert_bailingmoe_to_hf,
    "bailing_moe_linear": convert_bailingmoe_to_hf,
    "bailing_hybrid": convert_bailingmoe_to_hf,
}


def convert_to_hf(
    tf_config: TransformerConfig,
    model_name: str,
    name: str,
    param: Parameter | Tensor | FP8BlockwiseTensorHelper,
    quantization_config: dict[str, int | str | list[str]] | None = None,
    fp8_direct_convert: bool = False,
    hf_config=None,
):
    """Convert Megatron parameter to HuggingFace format, optionally with FP8 quantization.

    Args:
        tf_config: Transformer configuration
        model_name: Model name (e.g., "qwen2", "qwen3_moe")
        name: Parameter name in Megatron format
        param: Parameter tensor or FP8BlockwiseTensorHelper
        quantization_config: Optional quantization config dict with keys:
            - quant_method: "fp8"
            - fmt: "e4m3"
            - activation_scheme: "dynamic"
            - weight_block_size: Optional tuple/list of [block_m, block_n] for blockwise quantization
        fp8_direct_convert: If True, directly convert TE FP8 tensors to PyTorch FP8 format.
            If False, dequantize TE FP8 to bf16 first, then quantize to PyTorch FP8.
        hf_config: Optional HuggingFace PretrainedConfig. Required for VLM models
            that need vision_config for weight conversion (e.g., vision QKV reordering).

    Returns:
        List of (name, tensor) tuples in HuggingFace format. For FP8 quantization,
        returns both quantized weight and scale tensors.
    """
    for key, conversion_fn in _CONVERSION_FN_REGISTRY.items():
        if key in model_name:
            # Pass hf_config to converters that accept it (e.g., VLM models)
            if _accepts_hf_config(conversion_fn):
                converted_named_tensors = conversion_fn(
                    tf_config, name, param, hf_config=hf_config
                )
            else:
                converted_named_tensors = conversion_fn(tf_config, name, param)
            if quantization_config:
                if fp8_direct_convert:
                    return convert_fp8_helper_to_pytorch_fp8(converted_named_tensors)
                else:
                    # Quantize from bf16 to PyTorch FP8
                    return quantize_params(
                        name, converted_named_tensors, quantization_config
                    )
            return converted_named_tensors

    raise ValueError(f"Unsupported model for HF conversion: {model_name}")


def get_named_parameters(model_module, num_experts):
    def _iter_single(single_module):
        ep_size = mpu.get_expert_model_parallel_world_size()
        ep_rank = mpu.get_expert_model_parallel_rank()
        if num_experts:
            expert_offset = ep_rank * num_experts // ep_size
        else:
            expert_offset = 0

        config = getattr(single_module, "config", None)
        if config is None and hasattr(single_module, "module"):
            config = getattr(single_module.module, "config", None)
        if config is None:
            raise AttributeError("Megatron module does not expose transformer config")

        vp_stage = getattr(single_module, "virtual_pipeline_model_parallel_rank", None)
        if vp_stage is None and hasattr(single_module, "module"):
            vp_stage = getattr(
                single_module.module, "virtual_pipeline_model_parallel_rank", None
            )
        if vp_stage is None:
            try:
                vp_stage = mpu.get_virtual_pipeline_model_parallel_rank()
            except AssertionError:
                vp_stage = None

        layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for name, param in single_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            # Match either text-only (module.module.decoder.layers.X) or VLM
            # (module.module.language_model.decoder.layers.X) — both share the
            # same layer_offset (here `config` is the language_model config).
            # Vision blocks (module.module.vision_model.decoder.layers.X) are
            # intentionally not matched: vision is not PP-split.
            decoder_layers_pattern = (
                r"module\.module\.(language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            )
            match = re.match(decoder_layers_pattern, name)
            if not match:
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # mtp layer starts from layer 0
                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer.mlp.experts\.(.+)\.weight(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
                continue

            lm_prefix, layer_idx, rest = match.groups()
            lm_prefix = lm_prefix or ""
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield (
                    f"module.module.{lm_prefix}decoder.layers.{layer_idx}.mlp.experts.{rest}.weight{expert_idx}",
                    param,
                )
            else:
                yield (
                    f"module.module.{lm_prefix}decoder.layers.{layer_idx}.{rest}",
                    param,
                )

        # treat expert bias as normal parameters
        for name, buffer in single_module.named_buffers():
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = (
                r"module\.module\.(language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            )
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                lm_prefix, layer_idx, rest = match.groups()
                lm_prefix = lm_prefix or ""
                layer_idx = int(layer_idx) + layer_offset
                yield (
                    f"module.module.{lm_prefix}decoder.layers.{layer_idx}.{rest}",
                    buffer,
                )

    if isinstance(model_module, (list, tuple)):
        try:
            vp_world = mpu.get_virtual_pipeline_model_parallel_world_size()
            original_vp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
        except AssertionError:
            original_vp_rank = None
            vp_world = None

        for vpp_rank, single_module in enumerate(model_module):
            if vp_world and vp_world > 1:
                mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            yield from _iter_single(single_module)

        if (
            vp_world
            and vp_world > 1
            and original_vp_rank is not None
            and original_vp_rank >= 0
        ):
            mpu.set_virtual_pipeline_model_parallel_rank(original_vp_rank)
        return

    yield from _iter_single(model_module)
