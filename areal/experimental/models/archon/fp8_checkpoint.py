# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import torch

from areal.utils.logging import getLogger

logger = getLogger("ArchonFP8Checkpoint")


def _get_scale_inv_keys(path: str) -> list[str]:
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        return [k for k in weight_map if k.endswith("_scale_inv")]

    safetensors_path = os.path.join(path, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors import safe_open

        with safe_open(safetensors_path, framework="pt") as f:
            all_keys = list(f.keys())
        return [k for k in all_keys if k.endswith("_scale_inv")]

    return []


def _detect_fp8_checkpoint(path: str) -> bool:
    """Return True if the HF checkpoint at *path* contains FP8 weights.

    Detection heuristic: look for ``*_scale_inv`` keys in
    ``model.safetensors.index.json``, which is the signature of blockwise
    FP8 checkpoints (DeepSeek-V3 / Qwen3-FP8 format).

    Returns False if the index file is absent (single-file checkpoint without
    FP8 keys, or non-HF format).
    """
    return len(_get_scale_inv_keys(path)) > 0


def _prepare_fp8_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
    path: str,
    block_size: int = 128,
    *,
    _cached_keys: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Prepare *hf_state_dict* for loading an FP8 checkpoint via DCP.

    DCP only loads keys that already exist in the state dict.  This function
    must be called **before** ``dcp.load()`` to:

    1. Change the weight placeholder dtype from BF16 → ``float8_e4m3fn``.
    2. Insert a ``float32`` placeholder for each ``*_scale_inv`` key.

    Args:
        hf_state_dict: HF-format state dict with BF16 placeholder tensors
            (produced by ``state_dict_adapter.to_hf()``).
        path: HF checkpoint directory (must contain
            ``model.safetensors.index.json``).
        block_size: Blockwise quantization block size (default 128).
        _cached_keys: Pre-detected scale_inv keys to avoid re-reading
            the index file (pass from ``_get_scale_inv_keys`` result).

    Returns:
        The same dict, mutated in-place and returned for convenience.
    """
    scale_inv_keys = (
        _cached_keys if _cached_keys is not None else _get_scale_inv_keys(path)
    )
    if not scale_inv_keys:
        return hf_state_dict

    logger.info(
        f"Preparing FP8 state dict: {len(scale_inv_keys)} scale_inv keys detected"
    )

    for scale_key in scale_inv_keys:
        weight_key = scale_key.replace("_scale_inv", "")
        if weight_key not in hf_state_dict:
            logger.warning(
                f"Scale key {scale_key!r} has no matching weight {weight_key!r} "
                "in state dict — skipping"
            )
            continue

        old_placeholder = hf_state_dict[weight_key]

        if old_placeholder.dim() != 2:
            raise ValueError(
                f"FP8 checkpoint loading requires 2D weight placeholders, "
                f"but {weight_key!r} has {old_placeholder.dim()}D shape "
                f"{tuple(old_placeholder.shape)}. "
                f"3D fused expert weights are not yet supported."
            )

        hf_state_dict[weight_key] = torch.empty_like(
            old_placeholder, dtype=torch.float8_e4m3fn
        )

        weight_shape = old_placeholder.shape
        scale_shape = (
            (weight_shape[0] + block_size - 1) // block_size,
            (weight_shape[1] + block_size - 1) // block_size,
        )
        hf_state_dict[scale_key] = torch.empty(
            scale_shape, dtype=torch.float32, device=old_placeholder.device
        )

    return hf_state_dict


def weight_dequant_cpu(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    dst_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Blockwise FP8 dequantization in pure PyTorch (CPU-compatible).

    Equivalent to the Megatron ``weight_dequant`` Triton kernel::

        bf16[i, j] = fp8[i, j].to(bf16) * scale_inv[i // block_size, j // block_size]

    Used as a fallback when the tensor is on CPU (e.g. DCP with cpu_offload).

    Args:
        x: FP8 weight tensor, shape ``(M, N)``.
        s: Scale-inverse tensor, shape ``(ceil(M/bs), ceil(N/bs))``.
        block_size: Block size (must match the quantization block size).
        dst_dtype: Output dtype.

    Returns:
        Dequantized weight tensor in *dst_dtype*, shape ``(M, N)``.
    """
    M, N = x.shape
    # Expand scale_inv to (M, N) by repeating each block entry.
    s_expanded = s.repeat_interleave(block_size, dim=0)[:M].repeat_interleave(
        block_size, dim=1
    )[:, :N]
    # Cast FP8 → float32 first, multiply, then cast to dst_dtype.
    # Using float32 intermediate avoids bf16 × f32 → f32 promotion surprises.
    return (x.to(torch.float32) * s_expanded.to(torch.float32)).to(dst_dtype)


def _dequant_dtensor(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize a single FSDP-sharded FP8 DTensor on its local shard.

    Only ``Shard(0)`` placements are supported.  TP/ETP column-sharded
    weights require per-dimension scale slicing that is not yet implemented;
    calling this function with such placements raises ``ValueError``.
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import (
        compute_local_shape_and_global_offset,
    )
    from torch.distributed.tensor.placement_types import Shard

    # TODO(agent): Implement per-column scale slicing for Shard(1) to
    #   support TP/ETP FP8 checkpoint loading. Requires slicing scale_inv
    #   along dim-1 with block-boundary alignment (mirrors the dim-0 logic
    #   below). Tracked as Phase 2 of FP8 support.
    for p in weight_fp8.placements:
        if isinstance(p, Shard) and p.dim != 0:
            raise ValueError(
                f"_dequant_dtensor only supports Shard(0) placements, "
                f"got {weight_fp8.placements}. Column-sharded (TP/ETP) FP8 "
                f"dequantization requires per-column scale slicing which is "
                f"not yet implemented."
            )

    local_fp8 = weight_fp8._local_tensor

    _, global_offset = compute_local_shape_and_global_offset(
        weight_fp8.shape, weight_fp8.device_mesh, weight_fp8.placements
    )
    start_row = global_offset[0]
    local_M = local_fp8.shape[0]
    block_start = start_row // block_size
    block_end = (start_row + local_M + block_size - 1) // block_size
    local_scale = scale_inv[block_start:block_end, :]

    # When the shard boundary is not block-aligned, pad the top so the
    # Triton kernel's `s[i // block_size]` indexing stays correct.
    offset = start_row % block_size
    if offset > 0:
        local_fp8 = torch.nn.functional.pad(local_fp8, (0, 0, offset, 0))

    if local_fp8.is_cuda:
        from areal.engine.megatron_utils.fp8.kernels import weight_dequant

        local_bf16 = weight_dequant(
            local_fp8.contiguous(),
            local_scale.to(local_fp8.device).contiguous(),
            block_size=block_size,
            dst_dtype=target_dtype,
        )
    else:
        local_bf16 = weight_dequant_cpu(
            local_fp8.contiguous(),
            local_scale.contiguous(),
            block_size=block_size,
            dst_dtype=target_dtype,
        )

    if offset > 0:
        local_bf16 = local_bf16[offset:]

    return DTensor.from_local(
        local_bf16,
        weight_fp8.device_mesh,
        weight_fp8.placements,
        shape=weight_fp8.shape,
        stride=weight_fp8.stride(),
    )


def dequant_fp8_state_dict(
    hf_state_dict: dict[str, torch.Tensor],
    target_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Dequantize all FP8 weights in *hf_state_dict* to *target_dtype*.

    Detects FP8 weights by their ``float8_e4m3fn`` dtype.  For each FP8
    weight, the matching ``*_scale_inv`` key is located, the weight is
    dequantized, and the scale key is removed.

    Dequantization backend is chosen automatically:
    - GPU tensor → Megatron ``weight_dequant`` Triton kernel.
    - CPU tensor → ``weight_dequant_cpu`` pure-PyTorch fallback.

    This function operates in HF-key space, **before** ``from_hf()`` converts
    keys to Archon format.  After this call the state dict only contains
    standard BF16 weight tensors; ``from_hf()`` can proceed as normal.

    Args:
        hf_state_dict: HF-format state dict loaded by ``dcp.load()``.
        target_dtype: Dtype to dequantize into (typically ``torch.bfloat16``).
        block_size: Quantization block size (default 128).

    Returns:
        The same dict, with FP8 weights replaced by *target_dtype* tensors and
        all ``*_scale_inv`` entries removed.

    Raises:
        KeyError: If a FP8 weight has no matching ``*_scale_inv`` key.
    """
    fp8_dtypes = {torch.float8_e4m3fn}

    fp8_keys = [
        k
        for k, v in hf_state_dict.items()
        if isinstance(v, torch.Tensor) and v.dtype in fp8_dtypes
    ]

    if not fp8_keys:
        return hf_state_dict

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None  # type: ignore[assignment, misc]

    logger.info(f"Dequantizing {len(fp8_keys)} FP8 weights → {target_dtype}")

    scale_keys_to_remove: list[str] = []
    for key in fp8_keys:
        scale_key = key + "_scale_inv"
        if scale_key not in hf_state_dict:
            raise KeyError(
                f"FP8 weight {key!r} has no matching scale key {scale_key!r}"
            )

        weight_fp8 = hf_state_dict[key]
        scale_inv = hf_state_dict[scale_key]

        if DTensor is not None and isinstance(scale_inv, DTensor):
            scale_inv = scale_inv.full_tensor()

        if weight_fp8.dim() != 2:
            raise ValueError(
                f"FP8 dequantization requires 2D weight tensors, but "
                f"{key!r} has {weight_fp8.dim()}D shape {tuple(weight_fp8.shape)}. "
                f"3D fused expert weights are not yet supported."
            )

        if DTensor is not None and isinstance(weight_fp8, DTensor):
            hf_state_dict[key] = _dequant_dtensor(
                weight_fp8, scale_inv, block_size, target_dtype
            )
        elif weight_fp8.is_cuda:
            from areal.engine.megatron_utils.fp8.kernels import weight_dequant

            hf_state_dict[key] = weight_dequant(
                weight_fp8.contiguous(),
                scale_inv.to(weight_fp8.device).contiguous(),
                block_size=block_size,
                dst_dtype=target_dtype,
            )
        else:
            hf_state_dict[key] = weight_dequant_cpu(
                weight_fp8,
                scale_inv,
                block_size=block_size,
                dst_dtype=target_dtype,
            )

        scale_keys_to_remove.append(scale_key)

    for k in scale_keys_to_remove:
        del hf_state_dict[k]

    logger.info(
        f"Dequantization complete: {len(scale_keys_to_remove)} scale keys removed"
    )
    return hf_state_dict
