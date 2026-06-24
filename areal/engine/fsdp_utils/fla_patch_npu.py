# SPDX-License-Identifier: Apache-2.0

# NPU-specific patches to enable the FLA fast GDN implementation using triton kernels

# Adapted from ModelScope Swift
# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import importlib
from typing import Any

import torch
import torch_npu
from torch import nn

from areal.utils import logging

logger = logging.getLogger("FLAPatchNPU")


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


def patch_qwen3_5_chunk_gated_delta_rule_with_mindspeed() -> None:
    try:
        from mindspeed.core.ssm.chunk_gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:
        logger.warning(
            "Failed to import embedded MindSpeed chunk_gated_delta_rule: %s", exc
        )
        raise

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
        patched_modules.append(module_name)

    if patched_modules:
        logger.info(
            "Patched Qwen3.5 chunk_gated_delta_rule to embedded MindSpeed implementation: %s.",
            ", ".join(patched_modules),
        )


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
