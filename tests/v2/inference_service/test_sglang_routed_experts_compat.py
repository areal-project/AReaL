# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areal.v2.inference_service.sglang import routed_experts_compat as compat


def test_resolve_routed_experts_cache_rows_uses_prefill_tokens_when_chunking_off():
    server_args = SimpleNamespace(
        chunked_prefill_size=-1,
        dp_size=1,
        max_prefill_tokens=512,
        piecewise_cuda_graph_max_tokens=None,
    )

    rows = compat.resolve_routed_experts_device_cache_rows(
        max_running_requests=16,
        server_args=server_args,
    )

    assert rows == 512


def test_normalize_routed_experts_server_args_disables_piecewise_graph_for_r3():
    server_args = SimpleNamespace(
        enable_return_routed_experts=True,
        disable_piecewise_cuda_graph=False,
    )

    changed = compat.normalize_routed_experts_server_args(server_args)

    assert changed is True
    assert server_args.disable_piecewise_cuda_graph is True


def test_normalize_routed_experts_server_args_keeps_non_r3_path_unchanged():
    server_args = SimpleNamespace(
        enable_return_routed_experts=False,
        disable_piecewise_cuda_graph=False,
    )

    changed = compat.normalize_routed_experts_server_args(server_args)

    assert changed is False
    assert server_args.disable_piecewise_cuda_graph is False


def test_token_cache_patch_handles_token_level_prefill_rows(monkeypatch):
    pytest.importorskip("sglang")

    from sglang.srt import server_args as sglang_server_args
    from sglang.srt.layers.moe import routed_experts_capturer

    cache_cls = routed_experts_capturer._RoutedExpertsDeviceCache
    saved_init = cache_cls.__init__
    saved_capture = cache_cls.capture_fwd_routed_experts
    saved_attrs = {
        name: getattr(cache_cls, name)
        for name in (
            compat._PATCH_ATTR,
            compat._ORIGINAL_INIT_ATTR,
            compat._ORIGINAL_CAPTURE_ATTR,
        )
        if hasattr(cache_cls, name)
    }

    try:
        if hasattr(cache_cls, compat._ORIGINAL_INIT_ATTR):
            cache_cls.__init__ = getattr(cache_cls, compat._ORIGINAL_INIT_ATTR)
        if hasattr(cache_cls, compat._ORIGINAL_CAPTURE_ATTR):
            cache_cls.capture_fwd_routed_experts = getattr(
                cache_cls,
                compat._ORIGINAL_CAPTURE_ATTR,
            )
        for name in (
            compat._PATCH_ATTR,
            compat._ORIGINAL_INIT_ATTR,
            compat._ORIGINAL_CAPTURE_ATTR,
        ):
            if hasattr(cache_cls, name):
                delattr(cache_cls, name)

        monkeypatch.setattr(
            sglang_server_args,
            "_global_server_args",
            SimpleNamespace(
                chunked_prefill_size=-1,
                dp_size=1,
                max_prefill_tokens=512,
                piecewise_cuda_graph_max_tokens=None,
            ),
        )

        assert compat.apply_sglang_routed_experts_token_cache_patch() is True
        cache = cache_cls(
            max_running_requests=16,
            num_hidden_layers=2,
            num_experts_per_tok=6,
            num_fused_shared_experts=0,
            device="cpu",
        )
        topk_ids = torch.arange(129 * 6, dtype=torch.int32).reshape(129, 6)

        cache.capture_fwd_routed_experts(layer_id=1, topk_ids=topk_ids)

        assert cache.buffer.shape == (512, 2, 6)
        torch.testing.assert_close(cache.buffer[:129, 1], topk_ids)
    finally:
        cache_cls.__init__ = saved_init
        cache_cls.capture_fwd_routed_experts = saved_capture
        for name in (
            compat._PATCH_ATTR,
            compat._ORIGINAL_INIT_ATTR,
            compat._ORIGINAL_CAPTURE_ATTR,
        ):
            if hasattr(cache_cls, name):
                delattr(cache_cls, name)
        for name, value in saved_attrs.items():
            setattr(cache_cls, name, value)
