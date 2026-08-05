# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import NamedTuple

import torch


class Qwen3MoeIndexRemapConfig(NamedTuple):
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    head_dim: int
    num_moe_experts: int | None = None
    expert_model_parallel_size: int = 1
    expert_model_parallel_rank: int = 0


class RemappedIndices(NamedTuple):
    name: str
    indices: torch.Tensor


class MCoreShardDirtyBitset(NamedTuple):
    name: str
    packed_bitset: torch.Tensor
    shape: tuple[int, ...]
    shard_start: int
    shard_numel: int


_BIT_OFFSET_LOOKUP: dict[str, torch.Tensor] = {}


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _bit_offset_lookup(device: torch.device) -> torch.Tensor:
    key = str(device)
    table = _BIT_OFFSET_LOOKUP.get(key)
    if table is None:
        rows = []
        for value in range(256):
            offsets = [bit for bit in range(8) if value & (1 << bit)]
            offsets.extend([8] * (8 - len(offsets)))
            rows.append(offsets)
        table = torch.tensor(rows, dtype=torch.uint8, device=device)
        _BIT_OFFSET_LOOKUP[key] = table
    return table


def _iter_packed_bitset_indices(
    packed: torch.Tensor,
    numel: int,
    *,
    chunk_bytes: int = 1 << 20,
) -> Iterable[torch.Tensor]:
    """Yield sorted int64 set-bit index chunks from a packed dirty bitset."""
    if chunk_bytes <= 0:
        raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed bitset must be torch.uint8, got {packed.dtype}")

    needed = (numel + 7) // 8
    flat = packed.reshape(-1)
    if flat.numel() < needed:
        raise ValueError(
            f"Packed bitset is too short for {numel} bits: "
            f"need {needed} bytes, got {flat.numel()}"
        )
    if numel == 0:
        return

    lookup: torch.Tensor | None = None
    shifts: torch.Tensor | None = None
    for byte_start in range(0, needed, chunk_bytes):
        byte_end = min(byte_start + chunk_bytes, needed)
        chunk = flat[byte_start:byte_end]
        nonzero_bytes = chunk.nonzero(as_tuple=False).squeeze(1)
        if nonzero_bytes.numel() == 0:
            continue

        if nonzero_bytes.numel() > chunk.numel() // 2:
            if shifts is None:
                shifts = torch.arange(8, dtype=torch.int16, device=packed.device).view(
                    1, 8
                )
            bits = (
                ((chunk.to(torch.int16).view(-1, 1) >> shifts) & 1)
                .to(torch.bool)
                .reshape(-1)
            )
            base = byte_start * 8
            valid_bits = min(bits.numel(), numel - base)
            local = bits[:valid_bits].nonzero(as_tuple=False).squeeze(1)
            if local.numel() > 0:
                yield local.to(torch.int64) + base
            continue

        if lookup is None:
            lookup = _bit_offset_lookup(flat.device)
        byte_values = chunk.index_select(0, nonzero_bytes).to(torch.long)
        offsets = lookup.index_select(0, byte_values)
        valid = offsets < 8
        byte_indices = nonzero_bytes + byte_start
        candidates = byte_indices.view(-1, 1).to(torch.long) * 8 + offsets.to(
            torch.long
        )
        valid &= candidates < numel
        selected = candidates[valid]
        if selected.numel() > 0:
            yield selected


def _append(
    output: dict[str, list[torch.Tensor]],
    name: str,
    indices: torch.Tensor,
) -> None:
    if indices.numel() > 0:
        output[name].append(indices.to(torch.int64))


def _flatten_grouped(output: dict[str, list[torch.Tensor]]) -> list[RemappedIndices]:
    result: list[RemappedIndices] = []
    for name, parts in output.items():
        if not parts:
            continue
        merged = torch.cat(parts, dim=0)
        order = torch.argsort(merged)
        result.append(RemappedIndices(name=name, indices=merged[order].to(torch.int32)))
    return result


def _layer_rest(name: str) -> tuple[str, str] | None:
    match = re.match(r"module\.module\.decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        return None
    return match.group(1), match.group(2)


def _expert_id(local_expert_idx: str, cfg: Qwen3MoeIndexRemapConfig) -> int:
    idx = int(local_expert_idx)
    ep_size = max(1, int(cfg.expert_model_parallel_size))
    if cfg.num_moe_experts is None or ep_size <= 1:
        return idx
    experts_per_partition = int(cfg.num_moe_experts) // ep_size
    if experts_per_partition <= 0:
        return idx
    if idx >= experts_per_partition:
        return idx
    return idx + int(cfg.expert_model_parallel_rank) * experts_per_partition


def _remap_row_split_2d(
    hf_names: tuple[str, str],
    dirty_indices: torch.Tensor,
    shape: tuple[int, ...],
) -> list[RemappedIndices]:
    if len(shape) != 2:
        raise ValueError(f"row split expects 2D shape, got {shape}")
    rows, cols = shape
    if rows % 2 != 0:
        raise ValueError(f"row split expects even row count, got {rows}")
    half = rows // 2
    idx = dirty_indices.to(torch.int64).reshape(-1)
    row = idx // cols
    col = idx % cols
    output: dict[str, list[torch.Tensor]] = defaultdict(list)

    first = row < half
    _append(output, hf_names[0], row[first] * cols + col[first])
    second = ~first
    _append(output, hf_names[1], (row[second] - half) * cols + col[second])
    return _flatten_grouped(output)


def _remap_qkv_weight(
    layer_idx: str,
    dirty_indices: torch.Tensor,
    shape: tuple[int, ...],
    cfg: Qwen3MoeIndexRemapConfig,
) -> list[RemappedIndices]:
    if len(shape) != 2:
        raise ValueError(f"qkv weight expects 2D shape, got {shape}")
    rows, cols = shape
    value_num_per_group = cfg.num_attention_heads // cfg.num_query_groups
    group_span = (value_num_per_group + 2) * cfg.head_dim
    if rows % group_span != 0:
        raise ValueError(
            f"qkv rows mismatch: rows={rows}, expected a multiple of {group_span}"
        )
    num_groups = rows // group_span
    if num_groups <= 0 or cfg.num_query_groups % num_groups != 0:
        raise ValueError(
            f"qkv group mismatch: local_groups={num_groups}, "
            f"total_groups={cfg.num_query_groups}"
        )
    if cols != cfg.hidden_size:
        raise ValueError(f"qkv hidden mismatch: cols={cols}, hidden={cfg.hidden_size}")

    idx = dirty_indices.to(torch.int64).reshape(-1)
    row = idx // cols
    col = idx % cols
    group = row // group_span
    offset = row % group_span
    slot = offset // cfg.head_dim
    head_offset = offset % cfg.head_dim

    output: dict[str, list[torch.Tensor]] = defaultdict(list)
    q_mask = slot < value_num_per_group
    q_row = ((group[q_mask] * value_num_per_group + slot[q_mask]) * cfg.head_dim) + (
        head_offset[q_mask]
    )
    _append(
        output,
        f"model.layers.{layer_idx}.self_attn.q_proj.weight",
        q_row * cols + col[q_mask],
    )

    k_mask = slot == value_num_per_group
    k_row = group[k_mask] * cfg.head_dim + head_offset[k_mask]
    _append(
        output,
        f"model.layers.{layer_idx}.self_attn.k_proj.weight",
        k_row * cols + col[k_mask],
    )

    v_mask = slot == value_num_per_group + 1
    v_row = group[v_mask] * cfg.head_dim + head_offset[v_mask]
    _append(
        output,
        f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        v_row * cols + col[v_mask],
    )
    return _flatten_grouped(output)


def _remap_qkv_bias(
    layer_idx: str,
    dirty_indices: torch.Tensor,
    shape: tuple[int, ...],
    cfg: Qwen3MoeIndexRemapConfig,
) -> list[RemappedIndices]:
    if len(shape) != 1:
        raise ValueError(f"qkv bias expects 1D shape, got {shape}")
    value_num_per_group = cfg.num_attention_heads // cfg.num_query_groups
    q_span = value_num_per_group * cfg.head_dim
    group_span = q_span + 2 * cfg.head_dim
    if shape[0] % group_span != 0:
        raise ValueError(
            f"qkv bias length mismatch: length={shape[0]}, "
            f"expected a multiple of {group_span}"
        )
    num_groups = shape[0] // group_span
    if num_groups <= 0 or cfg.num_query_groups % num_groups != 0:
        raise ValueError(
            f"qkv bias group mismatch: local_groups={num_groups}, "
            f"total_groups={cfg.num_query_groups}"
        )

    idx = dirty_indices.to(torch.int64).reshape(-1)
    group = idx // group_span
    offset = idx % group_span
    output: dict[str, list[torch.Tensor]] = defaultdict(list)

    q_mask = offset < q_span
    _append(
        output,
        f"model.layers.{layer_idx}.self_attn.q_proj.bias",
        group[q_mask] * q_span + offset[q_mask],
    )

    k_mask = (offset >= q_span) & (offset < q_span + cfg.head_dim)
    _append(
        output,
        f"model.layers.{layer_idx}.self_attn.k_proj.bias",
        group[k_mask] * cfg.head_dim + (offset[k_mask] - q_span),
    )

    v_mask = offset >= q_span + cfg.head_dim
    _append(
        output,
        f"model.layers.{layer_idx}.self_attn.v_proj.bias",
        group[v_mask] * cfg.head_dim + (offset[v_mask] - q_span - cfg.head_dim),
    )
    return _flatten_grouped(output)


def remap_qwen3_moe_mcore_indices_to_hf(
    name: str,
    dirty_indices: torch.Tensor,
    shape: tuple[int, ...],
    cfg: Qwen3MoeIndexRemapConfig,
) -> list[RemappedIndices]:
    """Map Qwen3-MoE mcore dirty flat indices into HF payload flat indices."""
    if dirty_indices.numel() == 0:
        return []

    if name == "module.module.embedding.word_embeddings.weight":
        return [
            RemappedIndices("model.embed_tokens.weight", dirty_indices.to(torch.int32))
        ]
    if name == "module.module.output_layer.weight":
        return [RemappedIndices("lm_head.weight", dirty_indices.to(torch.int32))]
    if name == "module.module.decoder.final_layernorm.weight":
        return [RemappedIndices("model.norm.weight", dirty_indices.to(torch.int32))]

    layer = _layer_rest(name)
    if layer is None:
        raise ValueError(f"Unsupported Qwen3-MoE mcore parameter name: {name}")
    layer_idx, rest = layer

    expert_match = re.match(r"mlp\.experts\.(.+)\.weight(\d+)", rest)
    if expert_match:
        expert_rest, local_expert_idx = expert_match.groups()
        expert_idx = _expert_id(local_expert_idx, cfg)
        if expert_rest == "linear_fc1":
            return _remap_row_split_2d(
                (
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                ),
                dirty_indices,
                shape,
            )
        if expert_rest == "linear_fc2":
            return [
                RemappedIndices(
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                    dirty_indices.to(torch.int32),
                )
            ]
        raise ValueError(f"Unsupported Qwen3-MoE expert parameter name: {name}")

    if rest == "self_attention.linear_qkv.weight":
        return _remap_qkv_weight(layer_idx, dirty_indices, shape, cfg)
    if rest == "self_attention.linear_qkv.bias":
        return _remap_qkv_bias(layer_idx, dirty_indices, shape, cfg)
    if rest == "self_attention.linear_proj.weight":
        return [
            RemappedIndices(
                f"model.layers.{layer_idx}.self_attn.o_proj.weight",
                dirty_indices.to(torch.int32),
            )
        ]
    if rest == "mlp.linear_fc1.weight":
        return _remap_row_split_2d(
            (
                f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.up_proj.weight",
            ),
            dirty_indices,
            shape,
        )
    if rest == "mlp.linear_fc2.weight":
        return [
            RemappedIndices(
                f"model.layers.{layer_idx}.mlp.down_proj.weight",
                dirty_indices.to(torch.int32),
            )
        ]
    direct_layer_names = {
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
    }
    if rest in direct_layer_names:
        return [
            RemappedIndices(
                f"model.layers.{layer_idx}.{direct_layer_names[rest]}",
                dirty_indices.to(torch.int32),
            )
        ]

    raise ValueError(f"Unsupported Qwen3-MoE mcore parameter name: {name}")


def remap_qwen3_moe_mcore_param_to_hf_names(
    name: str,
    cfg: Qwen3MoeIndexRemapConfig,
) -> tuple[str, ...]:
    """Return HF payload names structurally covered by one local mcore param."""
    if name == "module.module.embedding.word_embeddings.weight":
        return ("model.embed_tokens.weight",)
    if name == "module.module.output_layer.weight":
        return ("lm_head.weight",)
    if name == "module.module.decoder.final_layernorm.weight":
        return ("model.norm.weight",)

    layer = _layer_rest(name)
    if layer is None:
        raise ValueError(f"Unsupported Qwen3-MoE mcore parameter name: {name}")
    layer_idx, rest = layer

    expert_match = re.match(r"mlp\.experts\.(.+)\.weight(\d+)", rest)
    if expert_match:
        expert_rest, local_expert_idx = expert_match.groups()
        expert_idx = _expert_id(local_expert_idx, cfg)
        if expert_rest == "linear_fc1":
            return (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
            )
        if expert_rest == "linear_fc2":
            return (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
            )
        raise ValueError(f"Unsupported Qwen3-MoE expert parameter name: {name}")

    if rest == "self_attention.linear_qkv.weight":
        return (
            f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            f"model.layers.{layer_idx}.self_attn.k_proj.weight",
            f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        )
    if rest == "self_attention.linear_qkv.bias":
        return (
            f"model.layers.{layer_idx}.self_attn.q_proj.bias",
            f"model.layers.{layer_idx}.self_attn.k_proj.bias",
            f"model.layers.{layer_idx}.self_attn.v_proj.bias",
        )
    if rest == "self_attention.linear_proj.weight":
        return (f"model.layers.{layer_idx}.self_attn.o_proj.weight",)
    if rest == "mlp.linear_fc1.weight":
        return (
            f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"model.layers.{layer_idx}.mlp.up_proj.weight",
        )
    if rest == "mlp.linear_fc2.weight":
        return (f"model.layers.{layer_idx}.mlp.down_proj.weight",)

    direct_layer_names = {
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm.weight",
        "pre_mlp_layernorm.weight": "post_attention_layernorm.weight",
        "mlp.router.weight": "mlp.gate.weight",
        "mlp.router.expert_bias": "mlp.gate.e_score_correction_bias",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
    }
    if rest in direct_layer_names:
        return (f"model.layers.{layer_idx}.{direct_layer_names[rest]}",)

    raise ValueError(f"Unsupported Qwen3-MoE mcore parameter name: {name}")


def remap_qwen3_moe_mcore_bitset_to_hf(
    name: str,
    packed_bitset: torch.Tensor,
    shape: tuple[int, ...],
    cfg: Qwen3MoeIndexRemapConfig,
    *,
    numel: int | None = None,
    chunk_bytes: int = 1 << 20,
) -> list[RemappedIndices]:
    """Map a packed mcore dirty bitset into HF sparse flat indices.

    B2 fused dirty-bit kernels should produce a compact bitset, not a full
    mcore-space indices tensor. This adapter preserves the existing Qwen3-MoE
    remap semantics while expanding the bitset in chunks, so peak CPU memory is
    bounded by one index chunk plus the final HF sparse outputs.
    """
    expected_numel = _numel(shape)
    actual_numel = expected_numel if numel is None else numel
    if actual_numel != expected_numel:
        raise ValueError(
            f"bitset numel mismatch for {name}: numel={actual_numel}, "
            f"shape numel={expected_numel}"
        )

    output: dict[str, list[torch.Tensor]] = defaultdict(list)
    for dirty_indices in _iter_packed_bitset_indices(
        packed_bitset,
        actual_numel,
        chunk_bytes=chunk_bytes,
    ):
        for remapped in remap_qwen3_moe_mcore_indices_to_hf(
            name,
            dirty_indices,
            shape,
            cfg,
        ):
            output[remapped.name].append(remapped.indices)
    return _flatten_grouped(output)


def remap_qwen3_moe_mcore_shard_bitset_to_hf(
    name: str,
    packed_bitset: torch.Tensor,
    shape: tuple[int, ...],
    cfg: Qwen3MoeIndexRemapConfig,
    *,
    shard_start: int,
    shard_numel: int,
    chunk_bytes: int = 1 << 20,
) -> list[RemappedIndices]:
    """Map an optimizer-owned mcore shard dirty bitset into HF sparse indices.

    A fused optimizer dirty-bit implementation naturally sees only the local
    distributed-optimizer shard, not the full mcore parameter.  ``shard_start``
    anchors that local flat bitset in the full mcore flat parameter space before
    applying the same Qwen3-MoE mcore->HF layout remap as the full-bitset path.
    """
    expected_numel = _numel(shape)
    if shard_start < 0:
        raise ValueError(f"shard_start must be non-negative, got {shard_start}")
    if shard_numel < 0:
        raise ValueError(f"shard_numel must be non-negative, got {shard_numel}")
    shard_end = shard_start + shard_numel
    if shard_end > expected_numel:
        raise ValueError(
            f"shard range out of bounds for {name}: "
            f"[{shard_start}, {shard_end}) exceeds numel={expected_numel}"
        )

    output: dict[str, list[torch.Tensor]] = defaultdict(list)
    for local_dirty_indices in _iter_packed_bitset_indices(
        packed_bitset,
        shard_numel,
        chunk_bytes=chunk_bytes,
    ):
        dirty_indices = local_dirty_indices + shard_start
        for remapped in remap_qwen3_moe_mcore_indices_to_hf(
            name,
            dirty_indices,
            shape,
            cfg,
        ):
            output[remapped.name].append(remapped.indices)
    return _flatten_grouped(output)


def remap_qwen3_moe_mcore_shard_bitsets_to_hf_masks(
    shard_bitsets: Iterable[MCoreShardDirtyBitset],
    cfg: Qwen3MoeIndexRemapConfig,
    *,
    hf_names: Iterable[str] | None = None,
    device_by_hf_name: Mapping[str, torch.device] | None = None,
    chunk_bytes: int = 1 << 20,
    fill_all_hf_names: bool = True,
) -> dict[str, torch.Tensor]:
    """Merge mcore shard dirty bitsets into HF sparse-index masks.

    The returned mapping is directly consumable by ``DeltaTracker.encode`` as an
    external mask dictionary: values are sorted ``int32`` flat indices in each HF
    payload tensor.  Passing ``hf_names`` fills unchanged/covered payload tensors
    with empty ``int32`` masks so the tracker does not treat them as dense
    fallback.
    """
    output: dict[str, list[torch.Tensor]] = defaultdict(list)
    covered_names: set[str] = set()
    for shard in shard_bitsets:
        if not fill_all_hf_names:
            covered_names.update(
                remap_qwen3_moe_mcore_param_to_hf_names(shard.name, cfg)
            )
        for remapped in remap_qwen3_moe_mcore_shard_bitset_to_hf(
            shard.name,
            shard.packed_bitset,
            shard.shape,
            cfg,
            shard_start=shard.shard_start,
            shard_numel=shard.shard_numel,
            chunk_bytes=chunk_bytes,
        ):
            output[remapped.name].append(remapped.indices)

    devices = device_by_hf_name or {}
    masks: dict[str, torch.Tensor] = {}
    for name, parts in output.items():
        if not parts:
            continue
        merged = torch.cat(parts, dim=0)
        target_device = devices.get(name)
        if target_device is not None:
            merged = merged.to(target_device)
        order = torch.argsort(merged)
        masks[name] = merged[order].to(torch.int32)

    if hf_names is not None:
        if fill_all_hf_names:
            fill_names = hf_names
        else:
            fill_names = [name for name in hf_names if name in covered_names]
        for name in fill_names:
            if name not in masks:
                masks[name] = torch.empty(
                    0,
                    dtype=torch.int32,
                    device=devices.get(name, torch.device("cpu")),
                )
    return masks
