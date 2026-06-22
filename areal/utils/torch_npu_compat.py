# SPDX-License-Identifier: Apache-2.0

"""Compatibility shims for torch_npu API drift across versions.

Applied on import; a no-op when torch_npu is absent (CUDA/CPU hosts). Import
this BEFORE ``mindspeed.megatron_adaptor`` so the shims are in place before
MindSpeed touches the affected symbols.
"""


def _apply() -> None:
    try:
        import torch
        import torch_npu
    except ImportError:
        return

    # post3 relocated `unsupported_dtype` out of the top-level namespace, but
    # MindSpeed reads `torch_npu.unsupported_dtype` at import -- re-expose it.
    if not hasattr(torch_npu, "unsupported_dtype"):
        try:
            from torch_npu._init.registry.backend import unsupported_dtype as _ud
        except Exception:  # noqa: BLE001
            _ud = [
                torch.quint8,
                torch.quint4x2,
                torch.quint2x4,
                torch.qint32,
                torch.qint8,
            ]
        torch_npu.unsupported_dtype = list(_ud)


_apply()
