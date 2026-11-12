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
