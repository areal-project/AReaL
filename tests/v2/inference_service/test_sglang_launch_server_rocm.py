# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from areal.v2.inference_service.sglang.launch_server import (
    normalize_rocm_attention_backend,
)


@pytest.mark.parametrize(
    ("hip_version", "deterministic", "configured", "expected", "changed"),
    [
        ("7.15.0", False, "fa3", "aiter", True),
        ("7.15.0", True, "fa3", "triton", True),
        ("7.15.0", False, "triton", "triton", False),
        ("7.15.0", False, None, None, False),
        (None, False, "fa3", "fa3", False),
    ],
)
def test_normalize_rocm_attention_backend(
    monkeypatch, hip_version, deterministic, configured, expected, changed
):
    monkeypatch.setattr(torch.version, "hip", hip_version, raising=False)
    server_args = SimpleNamespace(
        attention_backend=configured,
        enable_deterministic_inference=deterministic,
    )

    assert normalize_rocm_attention_backend(server_args) is changed
    assert server_args.attention_backend == expected


def test_awex_entrypoint_replaces_fa3_on_rocm(monkeypatch):
    import runpy
    import sys

    import sglang.srt.server_args as sglang_server_args
    import sglang.srt.utils as sglang_utils

    monkeypatch.setattr(torch.version, "hip", "7.15.0", raising=False)
    server_args = SimpleNamespace(
        attention_backend="fa3",
        enable_deterministic_inference=False,
        skip_server_warmup=True,
    )
    monkeypatch.setattr(
        sglang_server_args, "prepare_server_args", lambda argv: server_args
    )
    launched = []
    # The real http_server initializes the GPU on import.
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.entrypoints.http_server",
        SimpleNamespace(launch_server=lambda args, **kwargs: launched.append(args)),
    )
    monkeypatch.setattr(sglang_utils, "kill_process_tree", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["sglang_plugin"])

    runpy.run_module("areal.engine.awex.plugins.sglang_plugin", run_name="__main__")

    assert launched == [server_args]
    assert server_args.attention_backend == "aiter"
