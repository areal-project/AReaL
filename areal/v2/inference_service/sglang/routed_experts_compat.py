# SPDX-License-Identifier: Apache-2.0

"""Compatibility patches for SGLang routed expert capture."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_areal_token_sized_device_cache_patch"
_ORIGINAL_INIT_ATTR = "_areal_original_init"
_ORIGINAL_CAPTURE_ATTR = "_areal_original_capture"


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_routed_experts_device_cache_rows(
    max_running_requests: int | None,
    server_args: Any,
) -> int:
    """Return rows needed to capture per-token routed experts.

    SGLang 0.5.10.post1 sizes this buffer mainly by max running requests when
    chunked prefill is disabled. Routed expert capture writes one row per token
    during prefill, so AReaL sizes the cache by the server's prefill token limit.
    """

    rows = _positive_int(max_running_requests) or 1

    dp_size = _positive_int(getattr(server_args, "dp_size", None)) or 1
    chunked_prefill_size = _positive_int(
        getattr(server_args, "chunked_prefill_size", None)
    )
    if chunked_prefill_size is not None:
        rows = max(rows, chunked_prefill_size * dp_size)

    max_prefill_tokens = _positive_int(getattr(server_args, "max_prefill_tokens", None))
    if max_prefill_tokens is not None:
        rows = max(rows, max_prefill_tokens)

    piecewise_cuda_graph_max_tokens = _positive_int(
        getattr(server_args, "piecewise_cuda_graph_max_tokens", None)
    )
    if piecewise_cuda_graph_max_tokens is not None:
        rows = max(rows, piecewise_cuda_graph_max_tokens)

    return rows


def normalize_routed_experts_server_args(server_args: Any) -> bool:
    """Disable the older piecewise graph path when that setting exists."""

    if not getattr(server_args, "enable_return_routed_experts", False):
        return False
    if not hasattr(server_args, "disable_piecewise_cuda_graph"):
        # SGLang 0.5.20 uses a different prefill graph configuration and its
        # capturer handles graph-local slices. Do not add a dead setting.
        return False
    if getattr(server_args, "disable_piecewise_cuda_graph", False):
        return False
    setattr(server_args, "disable_piecewise_cuda_graph", True)
    logger.info(
        "Disabled SGLang piecewise CUDA graph because routed expert capture is enabled."
    )
    return True


def apply_sglang_routed_experts_token_cache_patch() -> bool:
    """Patch SGLang routed expert device cache to be token-sized.

    Returns ``True`` when this call applied the patch and ``False`` when the
    current process was already patched.
    """

    from packaging.version import Version

    from areal.utils.pkg_version import get_version

    if Version(get_version("sglang")).base_version == "0.5.20":
        return _patch_v0520_routed_experts_cache()

    import torch
    from sglang.srt.layers.moe import routed_experts_capturer
    from sglang.srt.server_args import get_global_server_args

    cache_cls = routed_experts_capturer._RoutedExpertsDeviceCache
    if getattr(cache_cls, _PATCH_ATTR, False):
        return False

    original_init = cache_cls.__init__
    original_capture = cache_cls.capture_fwd_routed_experts

    def patched_init(
        self,
        max_running_requests: int,
        num_hidden_layers: int,
        num_experts_per_tok: int,
        num_fused_shared_experts: int,
        device: str,
    ) -> None:
        server_args = get_global_server_args()
        rows = resolve_routed_experts_device_cache_rows(
            max_running_requests=max_running_requests,
            server_args=server_args,
        )
        self.buffer = torch.zeros(
            (
                rows,
                num_hidden_layers,
                num_experts_per_tok + num_fused_shared_experts,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._finalize_allocation_log()

    def patched_capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        batch, _ = topk_ids.shape
        if batch > self.buffer.shape[0]:
            rows = max(batch, self.buffer.shape[0] * 2)
            old_buffer = self.buffer
            new_buffer = old_buffer.new_zeros((rows, *old_buffer.shape[1:]))
            new_buffer[: old_buffer.shape[0]] = old_buffer
            self.buffer = new_buffer
            logger.warning(
                "Expanded SGLang routed expert device cache from %s to %s rows.",
                old_buffer.shape[0],
                rows,
            )
        original_capture(self, layer_id, topk_ids)

    cache_cls.__init__ = patched_init
    cache_cls.capture_fwd_routed_experts = patched_capture
    setattr(cache_cls, _ORIGINAL_INIT_ATTR, original_init)
    setattr(cache_cls, _ORIGINAL_CAPTURE_ATTR, original_capture)
    setattr(cache_cls, _PATCH_ATTR, True)

    logger.info("Applied SGLang routed expert token-cache compatibility patch.")
    return True


def _patch_v0520_routed_experts_cache() -> bool:
    """Size 0.5.20's capturer before it creates device and gather buffers."""
    from sglang.srt.runtime_context import (
        get_schedule,
        max_prefill_buffer_tokens,
    )
    from sglang.srt.state_capturer.routed_experts import RoutedExpertsCapturer

    if getattr(RoutedExpertsCapturer, _PATCH_ATTR, False):
        return False

    original_init = RoutedExpertsCapturer.__init__

    def patched_init(
        self,
        model_config,
        num_tokens: int,
        max_running_requests: int,
        num_fused_shared_experts: int,
        device: str,
    ) -> None:
        schedule = get_schedule()
        prefill_rows = _positive_int(max_prefill_buffer_tokens()) or 0
        if not _positive_int(getattr(schedule, "chunked_prefill_size", None)):
            prefill_rows = max(
                prefill_rows,
                _positive_int(getattr(schedule, "max_prefill_tokens", None)) or 0,
            )
        # The native constructor multiplies by dp_size and also sizes the
        # scattered-A2A gather buffer from the resulting device cache.
        # Keep allocation outside capture so CUDA graphs retain valid pointers.
        required_per_dp = max(max_running_requests, prefill_rows)
        original_init(
            self,
            model_config,
            num_tokens,
            required_per_dp,
            num_fused_shared_experts,
            device,
        )

    RoutedExpertsCapturer.__init__ = patched_init
    setattr(RoutedExpertsCapturer, _ORIGINAL_INIT_ATTR, original_init)
    setattr(RoutedExpertsCapturer, _PATCH_ATTR, True)
    logger.info("Applied SGLang 0.5.20 routed expert cache compatibility patch.")
    return True
