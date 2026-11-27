# SPDX-License-Identifier: Apache-2.0
"""B300 GDN state continuity across DSpark verification and the next decode."""

import os
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("AREAL_B300_FLASHINFER_FP32_GDN") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 3),
    reason="Requires the B300 FP32 FlashInfer GDN rollout environment",
)
def test_dspark_verify_commit_then_decode():
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNKernel,
    )
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    fi, ref = FlashInferGDNKernel(), TritonGDNKernel()
    fi._alignment_fallback_kernel = Mock(spec=[])  # Exercise FlashInfer itself.
    batch, width, heads, value_heads, dim = 4, 9, 16, 32, 128
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(124)

    def random(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    slots = torch.tensor([3, 0, 2, -1], device=device, dtype=torch.int32)
    rows = torch.arange(batch, device=device, dtype=torch.int32)
    gate, bias = random(value_heads), random(value_heads)

    def inputs(n):
        return dict(
            q=random(1, batch * n, heads, dim),
            k=random(1, batch * n, heads, dim),
            v=random(1, batch * n, value_heads, dim),
            a=random(batch * n, value_heads),
            b=random(batch * n, value_heads),
            A_log=gate,
            dt_bias=bias,
            cache_indices=slots,
            query_start_loc=rows.new_tensor(range(0, (batch + 1) * n, n)),
        )

    verify, decode = inputs(width), inputs(1)
    initial = random(batch + 1, value_heads, dim, dim, dtype=torch.float32) * 0.05
    accepted = rows.new_tensor([0, 3, 8])
    results = []
    for kernel in (fi, ref):
        state = initial.clone()
        snapshots = initial.new_zeros(batch, width, value_heads, dim, dim)
        output = kernel.target_verify(
            **verify,
            ssm_states=state,
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=rows,
            cache_steps=width,
            retrieve_parent_token=None,
        )
        assert torch.equal(state, initial)
        state[slots[:-1].long()] = snapshots[rows[:-1].long(), accepted.long()]
        following = kernel.decode(**decode, ssm_states=state)
        assert torch.equal(state[[1, 4]], initial[[1, 4]])
        results.append((output[:, :-width], snapshots[:-1], following[:, :-1], state))

    for actual, expected in zip(*results):
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=5e-3)


@pytest.mark.parametrize("algorithm", ["DSPARK", "AREAL_DSPARK"])
@pytest.mark.skipif(
    os.environ.get("AREAL_B300_FLASHINFER_FP32_GDN") != "1"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 3),
    reason="Requires the B300 FP32 FlashInfer GDN rollout environment",
)
def test_fixed_width_dspark_accepts_fp32_at_startup(monkeypatch, algorithm):
    from types import SimpleNamespace

    from sglang.srt.arg_groups import attention_hook
    from sglang.srt.server_args import ServerArgs

    monkeypatch.setenv("SGLANG_RAGGED_VERIFY_MODE", "static")
    monkeypatch.setattr(
        attention_hook,
        "model_config_of",
        lambda _: SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_5_moe")),
    )
    args = ServerArgs(
        model_path="dummy",
        device="cuda",
        speculative_algorithm=algorithm,
        linear_attn_decode_backend="flashinfer",
        linear_attn_verify_backend="flashinfer",
        mamba_ssm_dtype="float32",
    )
    attention_hook.handle_linear_attn_backend(args)
