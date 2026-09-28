# SPDX-License-Identifier: Apache-2.0

# NPU-specific patches to enable the FLA fast GDN implementation using triton kernels

# Adapted from ModelScope Swift
# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib
import os
from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from torch import nn

from areal.models.fsdp.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_world_size,
)
from areal.utils import logging

logger = logging.getLogger("FLAPatchNPU")

_NPU_GDN_CAUSAL_CONV1D = None


def import_optional_module(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        logger.debug("Failed to import optional module %s: %s", module_name, exc)
        return None


def apply_patch_map(root: Any, patch_map: dict[str, Any]) -> None:
    for path, value in patch_map.items():
        current = root
        parts = path.split(".")
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], value)


def _resolve_unsqueeze_dim(position_ids=None, unsqueeze_dim=1):
    if isinstance(position_ids, int) and unsqueeze_dim == 1:
        return position_ids
    return unsqueeze_dim


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    unsqueeze_dim = _resolve_unsqueeze_dim(position_ids, unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def npu_swiglu_forward(self, hidden_state):
    return self.down_proj(
        torch_npu.npu_swiglu(
            torch.cat(
                (self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1
            ),
            dim=-1,
        )
    )


class NpuQwen3_5RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        scale = (1.0 + self.weight).to(dtype=x.dtype)
        return torch_npu.npu_rms_norm(x, scale, epsilon=self.eps)[0]

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


def npu_apply_rotary_pos_emb_qwen3_5(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
):
    unsqueeze_dim = _resolve_unsqueeze_dim(position_ids, unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_rot = torch_npu.npu_rotary_mul(q_rot, cos, sin)
    k_rot = torch_npu.npu_rotary_mul(k_rot, cos, sin)

    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


def _patch_transformers_flash_linear_attention_available() -> None:
    def _is_flash_linear_attention_available() -> bool:
        return True

    transformers_utils = import_optional_module("transformers.utils")
    if transformers_utils is not None:
        setattr(
            transformers_utils,
            "is_flash_linear_attention_available",
            _is_flash_linear_attention_available,
        )

    transformers_import_utils = import_optional_module(
        "transformers.utils.import_utils"
    )
    if transformers_import_utils is not None:
        setattr(
            transformers_import_utils,
            "is_flash_linear_attention_available",
            _is_flash_linear_attention_available,
        )


def _cache_has_previous_state(cache_params: Any) -> bool:
    has_previous_state = getattr(cache_params, "has_previous_state", False)
    return bool(
        has_previous_state() if callable(has_previous_state) else has_previous_state
    )


def _update_cache_state(
    cache_params: Any, layer_idx: int, state_name: str, value: Any
) -> None:
    update_name = (
        f"update_{state_name[:-1] if state_name.endswith('s') else state_name}"
    )
    update_fn = getattr(cache_params, update_name, None)
    if callable(update_fn):
        update_fn(layer_idx, value)
        return
    getattr(cache_params, state_name)[layer_idx] = value


def _get_gdn_cu_seqlens(
    kwargs: dict[str, Any], device: torch.device
) -> torch.Tensor | None:
    cu_seqlens = kwargs.get("gdn_cu_seqlens")
    if cu_seqlens is None:
        cu_seqlens = kwargs.get("cu_seqlens")
    if cu_seqlens is None:
        cu_seqlens = kwargs.get("cu_seq_lens_q")
    if cu_seqlens is None:
        return None
    return cu_seqlens.reshape(-1).to(device=device, dtype=torch.int64)


def _maybe_append_padded_total(
    cu_seqlens: torch.Tensor | None, total_len: int
) -> torch.Tensor | None:
    if cu_seqlens is None or cu_seqlens.numel() == 0:
        return cu_seqlens
    final_len = int(cu_seqlens[-1].item())
    if final_len == total_len:
        return cu_seqlens
    if final_len > total_len:
        raise ValueError(
            f"cu_seqlens last value {final_len} exceeds GDN sequence length "
            f"{total_len}."
        )
    return torch.cat([cu_seqlens, cu_seqlens.new_tensor([total_len])], dim=0)


def _get_local_gdn_qkv_param(
    param: torch.Tensor | None,
    param_name: str,
    key_dim: int,
    value_dim: int,
    local_key_dim: int,
    local_value_dim: int,
    ulysses_rank: int,
) -> torch.Tensor | None:
    if param is None:
        return None
    expected_dim = key_dim * 2 + value_dim
    if param.shape[0] != expected_dim:
        raise ValueError(
            f"{param_name} dim ({param.shape[0]}) must match "
            f"(2 * key_dim + value_dim) ({expected_dim})."
        )
    k_off = ulysses_rank * local_key_dim
    v_off = ulysses_rank * local_value_dim
    q_param = param[k_off : k_off + local_key_dim]
    k_param = param[key_dim + k_off : key_dim + local_key_dim + k_off]
    v_param = param[2 * key_dim + v_off : 2 * key_dim + local_value_dim + v_off]
    return torch.cat([q_param, k_param, v_param], dim=0)


def _get_local_gdn_conv1d_weight(
    self, ulysses_rank: int, local_key_dim: int, local_value_dim: int
) -> torch.Tensor:
    if self.conv1d.weight.shape[0] == 1:
        return self.conv1d.weight
    return _get_local_gdn_qkv_param(
        self.conv1d.weight,
        "conv1d weight",
        self.key_dim,
        self.value_dim,
        local_key_dim,
        local_value_dim,
        ulysses_rank,
    )


def _get_local_gdn_conv1d_bias(
    self, ulysses_rank: int, local_key_dim: int, local_value_dim: int
) -> torch.Tensor | None:
    if self.conv1d.bias is None or self.conv1d.bias.shape[0] == 1:
        return self.conv1d.bias
    return _get_local_gdn_qkv_param(
        self.conv1d.bias,
        "conv1d bias",
        self.key_dim,
        self.value_dim,
        local_key_dim,
        local_value_dim,
        ulysses_rank,
    )


def _expand_shared_gdn_conv1d_weight(
    weight: torch.Tensor, local_dim: int
) -> torch.Tensor:
    if weight.shape[0] == local_dim:
        return weight
    if weight.shape[0] == 1:
        return weight.expand(local_dim, *weight.shape[1:]).contiguous()
    raise ValueError(
        f"conv1d weight dim ({weight.shape[0]}) must be 1 or local GDN dim "
        f"({local_dim})."
    )


def _expand_shared_gdn_conv1d_bias(
    bias: torch.Tensor | None, local_dim: int
) -> torch.Tensor | None:
    if bias is None or bias.shape[0] == local_dim:
        return bias
    if bias.shape[0] == 1:
        return bias.expand(local_dim).contiguous()
    raise ValueError(
        f"conv1d bias dim ({bias.shape[0]}) must be 1 or local GDN dim ({local_dim})."
    )


def _as_gdn_triton_conv1d_weight(weight: torch.Tensor, local_dim: int) -> torch.Tensor:
    if weight.ndim == 3:
        if weight.shape[0] == local_dim:
            return weight.squeeze(1).transpose(-1, -2).contiguous()
        if weight.shape[0] == 1:
            weight = weight.expand(local_dim, *weight.shape[1:])
            return weight.squeeze(1).transpose(-1, -2).contiguous()
    if weight.ndim == 2:
        if weight.shape[0] == local_dim:
            return weight.transpose(-1, -2).contiguous()
        if weight.shape[0] == 1:
            weight = weight.expand(local_dim, weight.shape[1])
            return weight.transpose(-1, -2).contiguous()
        if weight.shape[-1] == local_dim:
            return weight.contiguous()
    raise ValueError(
        "conv1d weight must be depthwise [D, 1, W]/[D, W], shared "
        f"[1, 1, W]/[1, W], or kernel [W, D], got {tuple(weight.shape)} "
        f"for local GDN dim {local_dim}."
    )


def _as_torch_conv1d_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 3:
        return weight
    if weight.ndim == 2:
        return weight.unsqueeze(1)
    raise ValueError(f"conv1d weight must be 2D or 3D, got {tuple(weight.shape)}.")


def qwen3_5_gated_deltanet_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params: Any | None = None,
    cache_position: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    module = importlib.import_module(self.__class__.__module__)
    apply_mask = getattr(module, "apply_mask_to_padding_states", None)
    if apply_mask is not None:
        hidden_states = apply_mask(hidden_states, attention_mask)
    elif attention_mask is not None and attention_mask.shape[1] > 1:
        hidden_states = hidden_states * attention_mask[:, :, None].to(
            hidden_states.dtype
        )

    batch_size, local_seq_len, _ = hidden_states.shape
    use_precomputed_states = (
        cache_params is not None
        and _cache_has_previous_state(cache_params)
        and local_seq_len == 1
        and cache_position is not None
    )

    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    z = self.in_proj_z(hidden_states).reshape(
        batch_size, local_seq_len, -1, self.head_v_dim
    )
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    ulysses_size = get_ulysses_sequence_parallel_world_size()
    use_ulysses = ulysses_size > 1
    if use_ulysses:
        ulysses_rank = get_ulysses_sequence_parallel_rank()
        if self.num_k_heads % ulysses_size != 0 or self.num_v_heads % ulysses_size != 0:
            raise ValueError(
                f"Ulysses SP size ({ulysses_size}) must divide num_k_heads "
                f"({self.num_k_heads}) and num_v_heads ({self.num_v_heads}) "
                "for Qwen3.5 GatedDeltaNet."
            )
        local_num_k_heads = self.num_k_heads // ulysses_size
        local_num_v_heads = self.num_v_heads // ulysses_size
        local_key_dim = self.head_k_dim * local_num_k_heads
        local_value_dim = self.head_v_dim * local_num_v_heads

        q_proj, k_proj, v_proj = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        q_proj = q_proj.reshape(
            batch_size, local_seq_len, self.num_k_heads, self.head_k_dim
        )
        k_proj = k_proj.reshape(
            batch_size, local_seq_len, self.num_k_heads, self.head_k_dim
        )
        v_proj = v_proj.reshape(
            batch_size, local_seq_len, self.num_v_heads, self.head_v_dim
        )
        q_proj = gather_seq_scatter_heads(q_proj, seq_dim=1, head_dim=2)
        k_proj = gather_seq_scatter_heads(k_proj, seq_dim=1, head_dim=2)
        v_proj = gather_seq_scatter_heads(v_proj, seq_dim=1, head_dim=2)

        b = b.reshape(batch_size, local_seq_len, self.num_v_heads)
        a = a.reshape(batch_size, local_seq_len, self.num_v_heads)
        b = gather_seq_scatter_heads(b, seq_dim=1, head_dim=2)
        a = gather_seq_scatter_heads(a, seq_dim=1, head_dim=2)

        mixed_qkv = torch.cat(
            [
                q_proj.reshape(batch_size, q_proj.shape[1], local_key_dim),
                k_proj.reshape(batch_size, k_proj.shape[1], local_key_dim),
                v_proj.reshape(batch_size, v_proj.shape[1], local_value_dim),
            ],
            dim=-1,
        )
    else:
        ulysses_rank = 0
        local_num_k_heads = self.num_k_heads
        local_num_v_heads = self.num_v_heads
        local_key_dim = self.key_dim
        local_value_dim = self.value_dim

    gdn_seq_len = mixed_qkv.shape[1]
    cu_seqlens = _maybe_append_padded_total(
        _get_gdn_cu_seqlens(kwargs, hidden_states.device), gdn_seq_len
    )

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        conv_weight = self.conv1d.weight
        conv_bias = self.conv1d.bias
        if use_ulysses:
            conv_weight = _get_local_gdn_conv1d_weight(
                self,
                ulysses_rank,
                local_key_dim,
                local_value_dim,
            )
            conv_bias = _get_local_gdn_conv1d_bias(
                self,
                ulysses_rank,
                local_key_dim,
                local_value_dim,
            )
        if cache_params is not None:
            conv_state = F.pad(
                mixed_qkv.transpose(1, 2),
                (self.conv_kernel_size - mixed_qkv.shape[1], 0),
            )
            _update_cache_state(cache_params, self.layer_idx, "conv_states", conv_state)

        causal_conv = _NPU_GDN_CAUSAL_CONV1D
        if causal_conv is not None:
            try:
                local_conv_dim = local_key_dim * 2 + local_value_dim
                triton_conv_weight = _as_gdn_triton_conv1d_weight(
                    conv_weight, local_conv_dim
                )
                triton_conv_bias = _expand_shared_gdn_conv1d_bias(
                    conv_bias, local_conv_dim
                )
                mixed_qkv, _ = causal_conv(
                    x=mixed_qkv,
                    weight=triton_conv_weight,
                    bias=triton_conv_bias,
                    activation=self.activation,
                    initial_state=None,
                    output_final_state=False,
                    cu_seqlens=cu_seqlens,
                )
            except NotImplementedError:
                local_conv_dim = local_key_dim * 2 + local_value_dim
                torch_conv_weight = _expand_shared_gdn_conv1d_weight(
                    conv_weight, local_conv_dim
                )
                torch_conv_bias = _expand_shared_gdn_conv1d_bias(
                    conv_bias, local_conv_dim
                )
                mixed_qkv_t = mixed_qkv.transpose(1, 2)
                mixed_qkv = F.conv1d(
                    mixed_qkv_t,
                    weight=_as_torch_conv1d_weight(torch_conv_weight),
                    bias=torch_conv_bias,
                    padding=self.conv_kernel_size - 1,
                    groups=local_conv_dim,
                )[:, :, :gdn_seq_len]
                mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)
        else:
            local_conv_dim = local_key_dim * 2 + local_value_dim
            torch_conv_weight = _expand_shared_gdn_conv1d_weight(
                conv_weight, local_conv_dim
            )
            torch_conv_bias = _expand_shared_gdn_conv1d_bias(conv_bias, local_conv_dim)
            mixed_qkv_t = mixed_qkv.transpose(1, 2)
            mixed_qkv = F.conv1d(
                mixed_qkv_t,
                weight=_as_torch_conv1d_weight(torch_conv_weight),
                bias=torch_conv_bias,
                padding=self.conv_kernel_size - 1,
                groups=local_conv_dim,
            )[:, :, :gdn_seq_len]
            mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)

    query, key, value = torch.split(
        mixed_qkv, [local_key_dim, local_key_dim, local_value_dim], dim=-1
    )
    query = query.reshape(batch_size, gdn_seq_len, local_num_k_heads, self.head_k_dim)
    key = key.reshape(batch_size, gdn_seq_len, local_num_k_heads, self.head_k_dim)
    value = value.reshape(batch_size, gdn_seq_len, local_num_v_heads, self.head_v_dim)

    beta = b.sigmoid()
    if use_ulysses:
        v_head_offset = ulysses_rank * local_num_v_heads
        v_head_slice = slice(v_head_offset, v_head_offset + local_num_v_heads)
        g = -self.A_log[v_head_slice].float().exp() * F.softplus(
            a.float() + self.dt_bias[v_head_slice]
        )
    else:
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    if self.num_v_heads // self.num_k_heads > 1:
        repeats = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

    if not use_precomputed_states:
        try:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        except TypeError:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        _update_cache_state(
            cache_params, self.layer_idx, "recurrent_states", last_recurrent_state
        )

    if use_ulysses:
        core_attn_out = gather_heads_scatter_seq(core_attn_out, head_dim=2, seq_dim=1)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, local_seq_len, -1)
    return self.out_proj(core_attn_out)


def patch_qwen3_5_chunk_gated_delta_rule_with_mindspeed() -> None:
    global _NPU_GDN_CAUSAL_CONV1D
    try:
        # patch L2 norm before importing GDN
        import areal.engine.megatron_utils.triton_l2norm_patch  # noqa: F401, I001
        from mindspeed.core.ssm.chunk_gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        logger.warning(
            "Failed to import embedded MindSpeed chunk_gated_delta_rule: %s", exc
        )
        raise

    causal_conv1d = None
    if os.environ.get("AREAL_DISABLE_NPU_GDN_CAUSAL_CONV1D") != "1":
        try:
            from areal.engine.fsdp_utils.gdn_triton.causal_conv1d import causal_conv1d
        except ImportError as exc:
            logger.warning(
                "Failed to import vendored NPU GDN causal_conv1d: %s. "
                "Falling back to torch conv1d.",
                exc,
            )
        else:
            _NPU_GDN_CAUSAL_CONV1D = causal_conv1d

    patched_modules = []
    for module_name in (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
    ):
        module = import_optional_module(module_name)
        if module is None:
            continue

        setattr(module, "is_flash_linear_attention_available", lambda: True)
        setattr(module, "is_fast_path_available", True)
        # FLA's fused RMSNormGated initializes with torch.cuda.current_device(),
        # so keep the native Qwen3.5 torch implementation on NPU.
        setattr(module, "FusedRMSNormGated", None)
        setattr(module, "chunk_gated_delta_rule", chunk_gated_delta_rule)
        if causal_conv1d is not None:
            setattr(module, "causal_conv1d_fn", causal_conv1d)
        if hasattr(module, "Qwen3_5GatedDeltaNet"):
            module.Qwen3_5GatedDeltaNet.forward = qwen3_5_gated_deltanet_forward
        if hasattr(module, "Qwen3_5MoeGatedDeltaNet"):
            module.Qwen3_5MoeGatedDeltaNet.forward = qwen3_5_gated_deltanet_forward
        patched_modules.append(module_name)

    if patched_modules:
        logger.info(
            "Patched Qwen3.5 chunk_gated_delta_rule to embedded MindSpeed implementation: %s.",
            ", ".join(patched_modules),
        )
        if causal_conv1d is not None:
            logger.info("Patched Qwen3.5 GatedDeltaNet to MindSpeed-MM causal_conv1d.")


QWEN3_5_PATCHES = {
    "Qwen3_5RMSNorm": NpuQwen3_5RMSNorm,
    "apply_rotary_pos_emb": npu_apply_rotary_pos_emb_qwen3_5,
    "Qwen3_5MLP.forward": npu_swiglu_forward,
}


def _build_patch_map(
    root, patches: dict[str, object], optional_patches: dict[str, object] | None = None
):
    patch_map = dict(patches)
    for path, value in (optional_patches or {}).items():
        current = root
        for part in path.split("."):
            if not hasattr(current, part):
                break
            current = getattr(current, part)
        else:
            patch_map[path] = value
    return patch_map


_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    patch_groups = []

    modeling_qwen3_5 = import_optional_module(
        "transformers.models.qwen3_5.modeling_qwen3_5"
    )
    if modeling_qwen3_5 is not None:
        _patch_transformers_flash_linear_attention_available()
        patch_qwen3_5_chunk_gated_delta_rule_with_mindspeed()

    if modeling_qwen3_5 is not None:
        patch_groups.append(("qwen3_5", modeling_qwen3_5, QWEN3_5_PATCHES, {}))

    for _group_name, module, patches, optional_patches in patch_groups:
        apply_patch_map(module, _build_patch_map(module, patches, optional_patches))

    _APPLIED = True


apply()
