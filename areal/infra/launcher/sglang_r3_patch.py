# SPDX-License-Identifier: Apache-2.0

"""SGLang server-side monkey patches for AReaL's R3 (Router Replay).

When ``skip_tokenizer_init=True`` (forced by R3), SGLang's ``TokenizerManager``
receives raw ``torch.Tensor`` routed_experts instead of base64-encoded strings.
FastAPI's ``jsonable_encoder`` silently converts tensors to ``{}``, breaking
the client-side decoder.  This patch base64-encodes the tensor in-place before
serialization, matching the format that ``DetokenizerManager`` produces in the
non-skip path.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False
_ROUTED_EXPERTS_ATTRS = ("routed_experts", "output_routed_experts")


def _encode_routed_experts_for_wire(value):
    """Convert ``routed_experts`` to a base64 string for wire transport.

    Mirrors ``DetokenizerManager._extract_routed_experts``.  Accepts tensors,
    numpy arrays, or already-encoded strings; other types returned unchanged.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        import numpy as np
        import pybase64
        import torch
    except Exception:  # pragma: no cover - defensive
        return value

    if isinstance(value, torch.Tensor):
        arr = value.detach().to("cpu").contiguous().numpy()
    elif isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
    else:
        return value

    if arr.dtype != np.int32:
        arr = arr.astype(np.int32, copy=False)

    return pybase64.b64encode(arr.tobytes()).decode("utf-8")


def apply_sglang_r3_patch() -> bool:
    """Install the ``_handle_batch_output`` monkey patch.

    Returns ``True`` when the patch is installed (or was already
    installed).  Returns ``False`` when SGLang is unavailable in the
    current process (so the caller can gracefully skip).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    try:
        from sglang.srt.managers import tokenizer_manager as _tm
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "[R3] sglang.srt.managers.tokenizer_manager not importable; "
            "skipping R3 server patch. reason=%s",
            exc,
        )
        return False

    original = _tm.TokenizerManager._handle_batch_output

    def _handle_batch_output_r3(self, recv_obj):  # type: ignore[no-redef]
        # Pre-encode routed_experts tensors to base64 before FastAPI serialization.
        try:
            for attr_name in _ROUTED_EXPERTS_ATTRS:
                re_list = getattr(recv_obj, attr_name, None)
                if re_list is None:
                    continue
                encoded = [_encode_routed_experts_for_wire(v) for v in re_list]
                try:
                    setattr(recv_obj, attr_name, encoded)
                except Exception:
                    # Frozen dataclass fallback.
                    object.__setattr__(recv_obj, attr_name, encoded)
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "[R3] Failed to pre-encode routed_experts on server; "
                "falling through to unpatched behaviour."
            )
        return original(self, recv_obj)

    _tm.TokenizerManager._handle_batch_output = _handle_batch_output_r3
    _PATCH_APPLIED = True
    logger.info(
        "[R3] Installed sglang TokenizerManager._handle_batch_output "
        "base64 encoder patch for routed_experts "
        "(handles both routed_experts and output_routed_experts)."
    )
    return True
