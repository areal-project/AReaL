# SPDX-License-Identifier: Apache-2.0

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import torch
import torch.distributed as dist
from mbridge.core.bridge import Bridge
from megatron.core import parallel_state as mpu
from megatron.core.fp8_utils import is_float8tensor
from safetensors import safe_open

from areal.engine.core.model import lang_config
from areal.engine.megatron_utils.fp8 import (
    FP8BlockwiseTensorHelper,
    dequantize_params,
    get_block_size_from_config,
)
from areal.infra.platforms import current_platform
from areal.models.mcore.registry import unwrap_to_gpt_model
from areal.utils import logging

logger = logging.getLogger("HFLoader")


def _get_tp_slice(shape, dim, tp_rank, tp_size) -> tuple:
    size_per_tp = shape[dim] // tp_size
    res = [slice(None) for _ in range(dim)]
    res.append(slice(tp_rank * size_per_tp, (tp_rank + 1) * size_per_tp))
    return tuple(res)


def _get_shape(obj) -> list:
    """Get shape from either a tensor or PySafeSlice object."""
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    else:
        # PySafeSlice object
        return obj.get_shape()


def _merge_qkv_weights(
    hf_config,
    mcore_weights_name: str,
    hf_weights_safe_slice: list,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | FP8BlockwiseTensorHelper:
    """Merge Q, K, V weights into a single QKV weight tensor."""
    assert len(hf_weights_safe_slice) == 3
    text_cfg = lang_config(hf_config)
    num_key_value_heads = text_cfg.num_key_value_heads
    hidden_dim = text_cfg.hidden_size
    num_attention_heads = text_cfg.num_attention_heads
    head_dim = getattr(text_cfg, "head_dim", hidden_dim // num_attention_heads)
    group_dim = head_dim * num_attention_heads // num_key_value_heads
    q, k, v = hf_weights_safe_slice
    # q k v might be tp split
    real_num_key_value_heads = _get_shape(q)[0] // group_dim
    s = _get_tp_slice((real_num_key_value_heads * group_dim,), 0, tp_rank, tp_size)
    q = q[s].reshape(
        real_num_key_value_heads // tp_size,
        group_dim,
        -1,
    )
    s = _get_tp_slice((real_num_key_value_heads * head_dim,), 0, tp_rank, tp_size)
    k = k[s].reshape(real_num_key_value_heads // tp_size, head_dim, -1)
    v = v[s].reshape(real_num_key_value_heads // tp_size, head_dim, -1)
    out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]
    return torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()


def _load_fused_qkv_weight(
    hf_config,
    hf_weights_safe_slice: list,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Load fused QKV weight/bias with format conversion and TP slicing.

    HF stores fused QKV in concatenated format:
        [Q_all, K_all, V_all] along dim 0, i.e., [q0,...,qH, k0,...,kH, v0,...,vH].
    Megatron-core expects interleaved format:
        [H, 3, D] along dim 0, i.e., [q0,k0,v0, q1,k1,v1, ...].

    Handles both 2D weights [qkv_size, hidden] and 1D biases [qkv_size].
    """
    assert len(hf_weights_safe_slice) == 1
    x = hf_weights_safe_slice[0]
    x = x[:] if not isinstance(x, torch.Tensor) else x

    text_cfg = lang_config(hf_config)
    num_heads = text_cfg.num_attention_heads
    num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_heads)
    head_dim = x.shape[0] // (num_heads + 2 * num_kv_heads)
    is_bias = x.dim() == 1

    # num_kv_heads == num_heads (no GQA) for this path
    assert num_kv_heads == num_heads, (
        f"_load_fused_qkv_weight requires num_kv_heads == num_heads (no GQA), "
        f"got num_kv_heads={num_kv_heads}, num_heads={num_heads}"
    )

    if is_bias:
        # 1D bias: [Q_all(H*D), K_all(Kv*D), V_all(Kv*D)]
        q = x[: num_heads * head_dim].view(num_heads, head_dim)
        k = x[num_heads * head_dim : (num_heads + num_kv_heads) * head_dim].view(
            num_kv_heads, head_dim
        )
        v = x[(num_heads + num_kv_heads) * head_dim :].view(num_kv_heads, head_dim)
        x = torch.stack([q, k, v], dim=1)  # [H, 3, D]
        x = x.reshape(-1)  # [H*3*D]
        if tp_size > 1:
            heads_per_tp = num_heads // tp_size
            x = x.view(num_heads, 3 * head_dim)
            x = x[tp_rank * heads_per_tp : (tp_rank + 1) * heads_per_tp]
            x = x.reshape(-1)
    else:
        # 2D weight: [Q_all(H*D), K_all(Kv*D), V_all(Kv*D), hidden]
        hidden = x.shape[1]
        q = x[: num_heads * head_dim].view(num_heads, head_dim, hidden)
        k = x[num_heads * head_dim : (num_heads + num_kv_heads) * head_dim].view(
            num_kv_heads, head_dim, hidden
        )
        v = x[(num_heads + num_kv_heads) * head_dim :].view(
            num_kv_heads, head_dim, hidden
        )
        x = torch.stack([q, k, v], dim=1)  # [H, 3, D, hidden]
        x = x.reshape(-1, hidden)  # [H*3*D, hidden]
        if tp_size > 1:
            heads_per_tp = num_heads // tp_size
            x = x.view(num_heads, 3 * head_dim, hidden)
            x = x[tp_rank * heads_per_tp : (tp_rank + 1) * heads_per_tp]
            x = x.reshape(-1, hidden)

    return x.contiguous()


def _merge_gate_up_weights(
    hf_weights_safe_slice: list,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | FP8BlockwiseTensorHelper:
    """Merge gate_proj and up_proj into a single fc1 weight tensor."""
    assert len(hf_weights_safe_slice) == 2, len(hf_weights_safe_slice)
    gate, up = hf_weights_safe_slice
    # chunk 0 for TP split
    gate = gate[
        _get_tp_slice(_get_shape(gate), dim=0, tp_rank=tp_rank, tp_size=tp_size)
    ]
    up = up[_get_tp_slice(_get_shape(up), dim=0, tp_rank=tp_rank, tp_size=tp_size)]
    return torch.cat([gate, up], dim=0)


def _slice_moe_expert_weight(
    hf_weights_safe_slice: list,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | FP8BlockwiseTensorHelper:
    """Slice MoE expert weight along the appropriate dimension."""
    assert len(hf_weights_safe_slice) == 1
    x = hf_weights_safe_slice[0]
    shape = _get_shape(x)
    # dim 1 chunk
    partition_dim = 1
    return x[_get_tp_slice(shape, dim=partition_dim, tp_rank=tp_rank, tp_size=tp_size)]


def _parse_local_expert_idx(mcore_weights_name: str) -> int:
    """Extract the per-rank-local expert index from an mcore expert param name.

    mcore weight names for grouped MoE end with ``weight{idx}`` where ``idx`` is
    the rank-local expert index (0..num_experts_per_rank-1). The global expert
    index is ``idx + (num_moe_experts // ep_size) * ep_rank``.
    """
    return int(mcore_weights_name.split(".weight")[-1])


def _slice_moe_expert_fc1_stacked_gate_up(
    hf_weights_safe_slice: list,
    mcore_weights_name: str,
    num_moe_experts: int,
    ep_rank: int,
    ep_size: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Slice a per-expert ``linear_fc1.weight{idx}`` shard from a 3D stacked
    HF ``gate_up_proj`` of shape ``[E, hidden, 2*expert_dim]``.

    Used for Qwen3-VL-MoE-style MoE checkpoints whose HF format keeps experts
    grouped under one tensor (vs Qwen3-MoE which exposes per-expert flat keys).
    Mirrors mbridge ``Qwen3VBaseBridge._weight_to_mcore_format`` MoE branch
    (``base_bridge.py:322-331``) plus AReaL's TP slicing of the gate/up halves.
    """
    assert len(hf_weights_safe_slice) == 1
    stacked = hf_weights_safe_slice[0]
    local_idx = _parse_local_expert_idx(mcore_weights_name)
    num_experts_per_rank = num_moe_experts // ep_size
    global_idx = local_idx + num_experts_per_rank * ep_rank
    expert = stacked[global_idx]  # [hidden, 2*expert_dim]
    if not isinstance(expert, torch.Tensor):
        expert = expert[:]
    expert_t = expert.T.contiguous()  # [2*expert_dim, hidden]
    gate, up = expert_t.chunk(2, dim=0)  # each [expert_dim, hidden]
    gate = gate[
        _get_tp_slice(_get_shape(gate), dim=0, tp_rank=tp_rank, tp_size=tp_size)
    ]
    up = up[_get_tp_slice(_get_shape(up), dim=0, tp_rank=tp_rank, tp_size=tp_size)]
    return torch.cat([gate, up], dim=0)


def _slice_moe_expert_fc2_stacked_down(
    hf_weights_safe_slice: list,
    mcore_weights_name: str,
    num_moe_experts: int,
    ep_rank: int,
    ep_size: int,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Slice a per-expert ``linear_fc2.weight{idx}`` shard from a 3D stacked
    HF ``down_proj`` of shape ``[E, expert_dim, hidden]``.

    After expert slice + transpose the result is ``[hidden, expert_dim]``,
    matching mcore's per-expert ``linear_fc2`` layout. TP shards along dim 1
    (input/expert_dim axis), the same axis ``_slice_moe_expert_weight`` uses.
    """
    assert len(hf_weights_safe_slice) == 1
    stacked = hf_weights_safe_slice[0]
    local_idx = _parse_local_expert_idx(mcore_weights_name)
    num_experts_per_rank = num_moe_experts // ep_size
    global_idx = local_idx + num_experts_per_rank * ep_rank
    expert = stacked[global_idx]  # [expert_dim, hidden]
    if not isinstance(expert, torch.Tensor):
        expert = expert[:]
    expert_t = expert.T.contiguous()  # [hidden, expert_dim]
    return expert_t[
        _get_tp_slice(_get_shape(expert_t), dim=1, tp_rank=tp_rank, tp_size=tp_size)
    ]


def _slice_generic_weight(
    mcore_param_shape: list,
    hf_weights_safe_slice: list,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor | FP8BlockwiseTensorHelper:
    """Slice generic weight tensor based on shape mismatch."""
    assert len(hf_weights_safe_slice) == 1
    x = hf_weights_safe_slice[0]
    x_shape = _get_shape(x)
    partition_dim = None
    if mcore_param_shape == x_shape:
        return x[:] if not isinstance(x, torch.Tensor) else x
    else:
        assert len(x_shape) == len(mcore_param_shape)
        for dim, (s1, s2) in enumerate(zip(x_shape, mcore_param_shape)):
            if s1 != s2:
                partition_dim = dim
                break
        # chunk on `partition_dim`
        return x[
            _get_tp_slice(x_shape, dim=partition_dim, tp_rank=tp_rank, tp_size=tp_size)
        ]


def _convert_vision_qkv_hf_to_mcore(
    hf_config,
    mcore_weights_name: str,
    mcore_param_shape: list,
    hf_weights_safe_slice: list,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Convert vision encoder QKV from HF format to mcore format.

    HF format: grouped [Q_all | K_all | V_all] (3 sections of num_heads*head_dim rows).
    mcore format: per-head interleaved [head_0(q,k,v) | head_1(q,k,v) | ...].
    Both have the same total shape but different internal ordering.

    Mirrors mbridge Qwen2_5VLBridge._weight_to_mcore_format for vision QKV.
    """
    # If 3 separate Q, K, V tensors, concatenate first into HF grouped format
    if len(hf_weights_safe_slice) == 3:
        parts = [
            w[:] if not isinstance(w, torch.Tensor) else w
            for w in hf_weights_safe_slice
        ]
        x = torch.cat(parts, dim=0)
    else:
        x = hf_weights_safe_slice[0]
        x = x[:] if not isinstance(x, torch.Tensor) else x

    vision_config = getattr(hf_config, "vision_config", None)
    if vision_config is None:
        # No vision_config means no special conversion is needed
        return _slice_generic_weight(mcore_param_shape, [x], tp_rank, tp_size)

    num_heads = vision_config.num_heads
    hidden_size = vision_config.hidden_size
    head_dim = hidden_size // num_heads
    is_bias = ".bias" in mcore_weights_name

    # Reshape from HF grouped format to mcore per-head interleaved format.
    # HF: [3, num_heads, head_dim, hidden_size] (weight) or [3, num_heads, head_dim] (bias)
    # mcore: [num_heads, 3*head_dim, hidden_size] (weight) or [num_heads, 3*head_dim] (bias)
    in_shape = (
        [3, num_heads, -1, head_dim, hidden_size] if not is_bias else [3, num_heads, -1]
    )
    q, k, v = x.view(*in_shape)

    head_shape = [num_heads, head_dim, -1] if not is_bias else [num_heads, head_dim]
    q = q.view(*head_shape)
    k = k.view(*head_shape)
    v = v.view(*head_shape)

    out_shape = [-1, hidden_size] if not is_bias else [-1]
    fused = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()

    return _slice_generic_weight(mcore_param_shape, [fused], tp_rank, tp_size)


def _weight_to_mcore_tp(
    hf_config,
    mcore_weights_name: str,
    mcore_param_shape: list,
    hf_weights_safe_slice: list,
    tp_rank: int,
    tp_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | FP8BlockwiseTensorHelper:
    """Convert HF weights to Megatron-Core format with tensor/expert parallelism.

    Dispatches to specialized handlers based on weight type:
    - QKV weights: merge Q, K, V into single tensor
    - FC1 weights: merge gate and up projections
    - MoE expert weights: slice along expert dimension
    - Generic weights: slice based on shape mismatch
    """
    if (
        "self_attention.linear_qkv." in mcore_weights_name
        and "layer_norm" not in mcore_weights_name
    ):
        if (
            len(hf_weights_safe_slice) == 3
            and "vision_model." not in mcore_weights_name
        ):
            res = _merge_qkv_weights(
                hf_config, mcore_weights_name, hf_weights_safe_slice, tp_rank, tp_size
            )
        elif "vision_model." in mcore_weights_name:
            # Vision encoder QKV: no GQA (num_heads == num_kv_heads). HF stores
            # QKV in grouped layout [Q_all | K_all | V_all] while mcore uses
            # per-head interleaved [head_0(q,k,v) | head_1(q,k,v) | ...].
            # Must convert between these formats (same shape, different ordering).
            res = _convert_vision_qkv_hf_to_mcore(
                hf_config,
                mcore_weights_name,
                mcore_param_shape,
                hf_weights_safe_slice,
                tp_rank,
                tp_size,
            )
        else:
            text_cfg = lang_config(hf_config)
            num_kv_heads = getattr(
                text_cfg, "num_key_value_heads", text_cfg.num_attention_heads
            )
            if num_kv_heads == text_cfg.num_attention_heads:
                # Fused QKV weight (e.g., Lightning Attention query_key_value)
                # Already in megatron interleaved format [H, 3, D] — just TP-slice
                res = _load_fused_qkv_weight(
                    hf_config, hf_weights_safe_slice, tp_rank, tp_size
                )
            else:
                # Fused QKV with GQA (e.g., Qwen2.5-VL language model qkv_proj)
                # Split into separate Q, K, V then merge into mcore format.
                x = hf_weights_safe_slice[0]
                x = x[:] if not isinstance(x, torch.Tensor) else x
                num_heads = text_cfg.num_attention_heads
                head_dim = x.shape[0] // (num_heads + 2 * num_kv_heads)
                q = x[: num_heads * head_dim]
                k = x[num_heads * head_dim : (num_heads + num_kv_heads) * head_dim]
                v = x[(num_heads + num_kv_heads) * head_dim :]
                res = _merge_qkv_weights(
                    hf_config,
                    mcore_weights_name,
                    [q, k, v],
                    tp_rank,
                    tp_size,
                )
    elif (
        "linear_fc1.weight" in mcore_weights_name
        or "linear_fc1.bias" in mcore_weights_name
    ):
        if len(hf_weights_safe_slice) == 2:
            # SwiGLU: merge separate gate_proj + up_proj
            res = _merge_gate_up_weights(hf_weights_safe_slice, tp_rank, tp_size)
        elif (
            "mlp.experts.linear_fc1.weight" in mcore_weights_name
            and len(hf_weights_safe_slice) == 1
            and len(_get_shape(hf_weights_safe_slice[0])) == 3
        ):
            # Stacked MoE expert (e.g., Qwen3-VL-MoE ``gate_up_proj`` shape
            # ``[E, hidden, 2*expert_dim]``). Slice per-expert + transpose +
            # gate/up TP-split. ``num_moe_experts`` lives on the bridge config.
            res = _slice_moe_expert_fc1_stacked_gate_up(
                hf_weights_safe_slice,
                mcore_weights_name,
                num_moe_experts=lang_config(hf_config).num_experts,
                ep_rank=mpu.get_expert_model_parallel_rank(),
                ep_size=mpu.get_expert_model_parallel_world_size(),
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        else:
            # Single fc1 weight (e.g., vision encoder MLP without gate/up split)
            res = _slice_generic_weight(
                mcore_param_shape, hf_weights_safe_slice, tp_rank, tp_size
            )
    elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:
        if (
            len(hf_weights_safe_slice) == 1
            and len(_get_shape(hf_weights_safe_slice[0])) == 3
        ):
            # Stacked MoE expert ``down_proj`` shape ``[E, expert_dim, hidden]``.
            res = _slice_moe_expert_fc2_stacked_down(
                hf_weights_safe_slice,
                mcore_weights_name,
                num_moe_experts=lang_config(hf_config).num_experts,
                ep_rank=mpu.get_expert_model_parallel_rank(),
                ep_size=mpu.get_expert_model_parallel_world_size(),
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        else:
            res = _slice_moe_expert_weight(hf_weights_safe_slice, tp_rank, tp_size)
    else:
        res = _slice_generic_weight(
            mcore_param_shape, hf_weights_safe_slice, tp_rank, tp_size
        )

    if dtype is not None and not isinstance(res, FP8BlockwiseTensorHelper):
        res = res.to(dtype)
    return res


def _load_weight_with_bridge_worker(
    bridge: Bridge,
    state_dict: dict[str, torch.Tensor],
    local_names: list[str],
    filenames: list[str],
    local_to_hf_map: dict[str, list[str]],
    weights_path: str,
    fp8_direct_convert: bool = False,
):
    all_slices = {}
    for filename in filenames:
        safetensor_file = os.path.join(weights_path, filename)
        with safe_open(safetensor_file, framework="pt", device="cpu") as f:
            for name in f.keys():
                all_slices[name] = f.get_slice(name)

    quantization_config = getattr(bridge.hf_config, "quantization_config", None)
    enable_fp8_param = (
        bridge.config.fp8 is not None and bridge.config.fp8_param and fp8_direct_convert
    )

    for local_name in local_names:
        hf_names = local_to_hf_map[local_name]
        param = state_dict[local_name]

        if "experts" in local_name and "shared_experts" not in local_name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
            tp_rank = mpu.get_expert_tensor_parallel_rank()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()

        # Get weight_block_size from quantization_config
        weight_block_size = get_block_size_from_config(quantization_config, strict=True)

        is_te_fp8_param = is_float8tensor(param)
        # Check if any HF weight is FP8 (has _scale_inv suffix)
        # If fp8 mode is not enabled in megatron,
        # we need to dequantize FP8 weights before converting to mcore format
        # Now only support FP8 dequantization
        hf_weights_safe_slice = []
        hf_has_fp8 = False
        hf_all_fp8 = True  # Track if all inputs are FP8

        for hf_name in hf_names:
            if "_scale_inv" in hf_name:
                continue
            hf_slice = all_slices[hf_name]
            scale_inv_name = f"{hf_name}_scale_inv"
            if scale_inv_name in all_slices:
                # HF weight is FP8
                hf_has_fp8 = True
                scale_inv_slice = all_slices[scale_inv_name]

                if is_te_fp8_param and enable_fp8_param:
                    # Convert to FP8BlockwiseTensorHelper to simplify handling
                    weight = hf_slice[:]
                    scale_inv = scale_inv_slice[:]
                    weight_helper = FP8BlockwiseTensorHelper(
                        weight, scale_inv, block_size=weight_block_size
                    )
                    hf_weights_safe_slice.append(weight_helper)
                else:
                    # Dequantize to higher precision (bf16)
                    device = torch.device(current_platform.device_type)
                    weight = hf_slice[:].to(device)
                    scale_inv = scale_inv_slice[:].to(device)
                    dequantized_weight = dequantize_params(
                        weight,
                        scale_inv,
                        dst_dtype=bridge.dtype,
                        quantization_config=quantization_config,
                    )
                    dequantized_weight = dequantized_weight.cpu()
                    hf_weights_safe_slice.append(dequantized_weight)
                    hf_all_fp8 = False
            else:
                hf_weights_safe_slice.append(hf_slice)
                hf_all_fp8 = False

        # If target is TE FP8 but not all inputs are FP8, we can't merge FP8 and non-FP8 tensors
        if is_te_fp8_param and enable_fp8_param and hf_has_fp8 and not hf_all_fp8:
            raise RuntimeError("Expected all inputs to be FP8 for TE FP8 parameter")

        param_to_load = _weight_to_mcore_tp(
            hf_config=bridge.hf_config,
            mcore_weights_name=local_name,
            mcore_param_shape=list(param.shape),
            hf_weights_safe_slice=hf_weights_safe_slice,
            tp_rank=tp_rank,
            tp_size=tp_size,
            dtype=bridge.dtype
            if not (is_te_fp8_param and hf_has_fp8 and hf_all_fp8)
            else None,
        )

        # Load the parameter
        if is_te_fp8_param and hf_has_fp8 and hf_all_fp8 and enable_fp8_param:
            # Direct FP8 to FP8 conversion
            try:
                from transformer_engine.pytorch.constants import TE_DType_To_Torch
            except ImportError as e:
                raise ImportError(
                    "transformer_engine is required for FP8 training. "
                    "Please install transformer_engine to use FP8 functionality."
                ) from e
            if TE_DType_To_Torch[param._fp8_dtype] is not param_to_load.dtype:
                raise ValueError(
                    f"Expected {TE_DType_To_Torch[param._fp8_dtype]} tensor for TE FP8 param, got {param_to_load.dtype}"
                )
            param_to_load.to_te_fp8_inplace(param)
        else:
            # NOTE: for megatron FP8 param, `param.copy_` will do quantization internally
            param.copy_(param_to_load, non_blocking=True)


def make_filename_bins(
    local_to_file_map: dict[str, list[str]],
) -> tuple[list[list[str]], list[list[str]]]:
    # Allocate local weight name into bins, where each bin access independent files
    # Then we can use multiple threads to concurrently load each bin's parameters.
    # This function has a complexity of O(F + L²)
    # where F = total number of files, L = number of local names
    if not local_to_file_map:
        return [], []

    local_names = list(local_to_file_map.keys())
    n = len(local_names)

    # Convert file lists to sets for O(1) lookups and create file-to-locals mapping
    local_to_files = {name: set(local_to_file_map[name]) for name in local_names}
    file_to_locals = defaultdict(set)
    for local_name, files in local_to_files.items():
        for file in files:
            file_to_locals[file].add(local_name)

    # Union-Find with path compression and union by rank
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x, y):
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            return

        # Union by rank
        if rank[root_x] < rank[root_y]:
            root_x, root_y = root_y, root_x
        parent[root_y] = root_x
        if rank[root_x] == rank[root_y]:
            rank[root_x] += 1

    # Create name-to-index mapping for O(1) lookups
    name_to_idx = {name: i for i, name in enumerate(local_names)}

    # Union locals that share files - O(F) where F is total number of files
    for locals_sharing_file in file_to_locals.values():
        if len(locals_sharing_file) > 1:
            locals_list = list(locals_sharing_file)
            first_idx = name_to_idx[locals_list[0]]
            for local_name in locals_list[1:]:
                union(first_idx, name_to_idx[local_name])

    # Group by root - O(L)
    root_to_group = defaultdict(list)
    for i, name in enumerate(local_names):
        root_to_group[find(i)].append(name)

    # Build result groups - O(L + F)
    grouped_local_names = []
    grouped_filenames = []

    for group in root_to_group.values():
        grouped_local_names.append(group)
        # Use set union to merge files from all locals in group
        all_files = set()
        for local_name in group:
            all_files.update(local_to_files[local_name])
        grouped_filenames.append(list(all_files))

    return grouped_local_names, grouped_filenames


def load_weights_from_hf_with_mbridge_fast(
    bridge: Bridge,
    models: list[torch.nn.Module],
    weights_path: str,
    max_workers: int | None = None,
    is_critic: bool = False,
    fp8_direct_convert: bool = False,
) -> None:
    weights_path = bridge._get_actual_hf_path(weights_path)
    index_file = os.path.join(weights_path, "model.safetensors.index.json")
    manual_tie_word_embedding = False
    index = {}
    if os.path.exists(index_file):
        with open(index_file, encoding="utf-8") as f:
            index = json.load(f)["weight_map"]
    else:
        # Search all safetensors files
        safetensor_files = glob(os.path.join(weights_path, "*.safetensors"))
        # If there are safetensors files
        if safetensor_files:
            # Iterate through each safetensors file
            for safetensor_file in safetensor_files:
                with safe_open(safetensor_file, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        index[k] = safetensor_file
        else:
            raise FileNotFoundError("No safetensors found in the model path to load.")
    if "model.embed_tokens.weight" in index and "lm_head.weight" not in index:
        manual_tie_word_embedding = True
        index["lm_head.weight"] = index["model.embed_tokens.weight"]

    # Calling model.state_dict() is very expensive
    # We call it in advance
    state_dicts = [model.state_dict() for model in models]

    worker_args = []
    tik = time.perf_counter()
    for model_index, model in enumerate(models):
        # map local weight names to global weight names
        local_to_global_map = bridge._weight_name_mapping_mcore_local_to_global(model)
        # map local weight names to huggingface weight names
        local_to_hf_map = {
            k: bridge._weight_name_mapping_mcore_to_hf(v)
            for k, v in local_to_global_map.items()
            if "_extra_state" not in k
        }
        if manual_tie_word_embedding:
            for k, v in local_to_hf_map.items():
                if "lm_head.weight" in v:
                    v.remove("lm_head.weight")
                    if "model.embed_tokens.weight" not in v:
                        v.append("model.embed_tokens.weight")

        local_to_file_map = defaultdict(list)
        for local_name, hf_names in local_to_hf_map.items():
            # Skip output_layer for critic models - it will be loaded separately
            if is_critic and "output_layer" in local_name:
                continue
            for name in hf_names:
                if "_scale_inv" in name:
                    continue
                filename = index[name]
                if filename not in local_to_file_map[local_name]:
                    local_to_file_map[local_name].append(filename)
                # Also include the scale_inv file if it exists
                scale_inv_name = f"{name}_scale_inv"
                if scale_inv_name in index:
                    scale_inv_filename = index[scale_inv_name]
                    if scale_inv_filename not in local_to_file_map[local_name]:
                        local_to_file_map[local_name].append(scale_inv_filename)

        grouped_local_names, grouped_filenames = make_filename_bins(local_to_file_map)

        for local_names, filenames in zip(grouped_local_names, grouped_filenames):
            worker_args.append(
                dict(
                    bridge=bridge,
                    state_dict=state_dicts[model_index],
                    local_names=local_names,
                    filenames=filenames,
                    local_to_hf_map=local_to_hf_map,
                    weights_path=weights_path,
                    fp8_direct_convert=fp8_direct_convert,
                )
            )

    logger.debug(
        f"Loading mcore weights from HF preparation time: {time.perf_counter() - tik}"
    )
    if max_workers is None:
        max_workers = min(8, max(1, os.cpu_count() // dist.get_world_size()))
    max_workers = min(max_workers, len(worker_args))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda kwargs: _load_weight_with_bridge_worker(**kwargs), worker_args
        )
        # Consume all results to make result all tasks complete
        for _ in results:
            pass

    # Load value_head weights for critic models.
    if is_critic and mpu.is_pipeline_last_stage():
        value_head_path = os.path.join(weights_path, "value_head.pt")
        if os.path.exists(value_head_path):
            value_head_state = torch.load(value_head_path, weights_only=True)
            for model in models:
                _model = unwrap_to_gpt_model(model)
                if hasattr(_model, "output_layer"):
                    _model.output_layer.load_state_dict(value_head_state)
            logger.info(f"Loaded ValueHead weights from {value_head_path}")
        else:
            logger.info(
                f"ValueHead checkpoint not found at {value_head_path}, "
                "using random initialization (normal for first training)."
            )
