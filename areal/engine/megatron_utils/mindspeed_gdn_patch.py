# SPDX-License-Identifier: Apache-2.0
"""Runtime fix-ups for MindSpeed's packed-capable GatedDeltaNet."""

from __future__ import annotations

import importlib
import weakref
from collections.abc import Sequence

import torch

from areal.utils import logging

logger = logging.getLogger("MegatronEngine")

# Pointer-keyed host copies of packed ``cu_seqlens``. MindSpeed's AscendC
# adapters otherwise call ``.detach().cpu().tolist()`` on a device tensor —
# a host-device stream stall — once per GDN layer. The values are known
# when the tensor is constructed, so no kernel has to be waited on.
_CU_HOST: dict[int, list[int]] = {}


def _try_import_module(name: str):
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError, AttributeError, KeyError):
        return None


def _read_cu_seqlens_host(cu: torch.Tensor) -> list[int]:
    if cu.device.type == "cpu":
        return [int(v) for v in cu.tolist()]
    return [int(v) for v in cu.detach().cpu().tolist()]


def _forget_cu_seqlens_ptr(ptr: int, host: list[int]) -> None:
    if _CU_HOST.get(ptr) is host:
        _CU_HOST.pop(ptr, None)


def remember_cu_seqlens(
    cu: torch.Tensor | None,
    host: Sequence[int] | None = None,
) -> torch.Tensor | None:
    """Record a Python ``list[int]`` next to ``cu``'s storage pointer.

    Call this when ``cu_seqlens`` is constructed (or moved to device) so
    MindSpeed's GDN adapters can read the values without a stream sync.
    """
    if cu is None or not torch.is_tensor(cu):
        return cu
    ptr = cu.data_ptr()
    if host is None:
        host = _CU_HOST.get(ptr)
    if host is None:
        host = _read_cu_seqlens_host(cu)
    host_list = host if isinstance(host, list) else [int(v) for v in host]
    _CU_HOST[ptr] = host_list
    weakref.finalize(cu, _forget_cu_seqlens_ptr, ptr, host_list)
    return cu


def cu_seqlens_host_list(cu: torch.Tensor) -> list[int]:
    """Return ``cu`` as ``list[int]``, preferring the cached host copy."""
    known = _CU_HOST.get(cu.data_ptr())
    if known is not None:
        return known
    host_list = _read_cu_seqlens_host(cu)
    remember_cu_seqlens(cu, host=host_list)
    return host_list


def _patch_prepare_packed_metadata(module: object) -> None:
    """Re-register MindSpeed's int64 ``cu.to(...)`` copy with the host list.

    ``npu_chunk_gated_delta_rule._prepare_packed_metadata`` does
    ``cu_seqlens.to(device=g.device, dtype=torch.int64)`` before
    ``_as_int_list``. That conversion allocates a new storage (new
    ``data_ptr``) every GDN layer, so the original cache would miss and
    each layer would stall again. Copy the known host list onto the
    converted tensor, then let the original helper run (its ``.to`` is
    then a no-op).
    """
    original = getattr(module, "_prepare_packed_metadata", None)
    if original is None or getattr(original, "_areal_known_cu_seqlens", False):
        return

    def wrapped(cu_seqlens, g, chunk_size):
        if cu_seqlens is not None:
            host = cu_seqlens_host_list(cu_seqlens)
            cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
            remember_cu_seqlens(cu_seqlens, host=host)
        return original(cu_seqlens, g, chunk_size)

    wrapped._areal_known_cu_seqlens = True
    module._prepare_packed_metadata = wrapped


def _known_cu_seqlens() -> bool:
    """Teach MindSpeed's two AscendC adapters to read cached host cu_seqlens.

    Both otherwise call ``.detach().cpu().tolist()`` on a device tensor,
    which blocks the host until the NPU stream drains: 72 stalls in one
    Qwen3.5 step (24 GDN mixers × conv + delta rule + its backward).
    Tensors that were never registered keep the device read.
    """
    npu_causal_conv1d = _try_import_module("mindspeed.core.ssm.npu_causal_conv1d")
    npu_chunk_gated_delta_rule = _try_import_module(
        "mindspeed.core.ssm.npu_chunk_gated_delta_rule"
    )
    patched = False
    if npu_causal_conv1d is not None and hasattr(npu_causal_conv1d, "_query_start_loc"):
        npu_causal_conv1d._query_start_loc = cu_seqlens_host_list
        patched = True
    if npu_chunk_gated_delta_rule is not None:
        if hasattr(npu_chunk_gated_delta_rule, "_as_int_list"):
            npu_chunk_gated_delta_rule._as_int_list = cu_seqlens_host_list
            patched = True
        _patch_prepare_packed_metadata(npu_chunk_gated_delta_rule)
    return patched


def has_mindspeed_gdn_model_classes(*gdn_classes: type) -> bool:
    """Return whether all captured GDN classes come from MindSpeed."""
    return all(cls.__module__.startswith("mindspeed") for cls in gdn_classes)


def has_mindspeed_gdn_conv1d(mindspeed_gdn: object) -> bool:
    """Return whether MindSpeed's GDN module has a causal conv implementation."""
    return getattr(mindspeed_gdn, "causal_conv1d", None) is not None


def ensure_mindspeed_gdn_conv1d() -> bool:
    """Bind the selected MindSpeed backend's causal conv into MCore GDN.

    Also installs ``_known_cu_seqlens`` so packed GDN kernels read host
    ``cu_seqlens`` from ``_CU_HOST`` instead of stalling the NPU stream.

    Returns True when a varlen conv implementation is available afterwards.
    """
    _known_cu_seqlens()
    try:
        mcore_gdn = importlib.import_module("megatron.core.ssm.gated_delta_net")
        ms_gdn = importlib.import_module("mindspeed.core.ssm.gated_delta_net")
    except ImportError:
        return False

    try:
        backend = importlib.import_module("mindspeed.core.ssm.gdn_backend")
        npu_conv = backend.load_gdn_operators().causal_conv1d
    except (AttributeError, ImportError):
        return False

    if (
        getattr(mcore_gdn, "causal_conv1d", None) is npu_conv
        and getattr(ms_gdn, "causal_conv1d", None) is npu_conv
    ):
        return True

    mcore_gdn.causal_conv1d = npu_conv
    ms_gdn.causal_conv1d = npu_conv
    logger.info("Activated the selected MindSpeed GDN causal_conv1d backend.")
    return True
