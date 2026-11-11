# SPDX-License-Identifier: Apache-2.0
"""AscendC GDN forward/backward against an independent recurrent reference."""

import importlib
import math
from unittest.mock import patch

import pytest
import torch

from areal.engine.megatron_utils.mindspeed_gdn_patch import (
    ensure_mindspeed_gdn_native_kkt,
)

pytest.importorskip("torch_npu")
pytest.importorskip("fla_npu")
pytestmark = [
    pytest.mark.npu,
    pytest.mark.slow,
    pytest.mark.ci,
    pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU"),
]

adapter = importlib.import_module("mindspeed.core.ssm.npu_chunk_gated_delta_rule")
native = importlib.import_module("fla_npu.ops.ascendc")


@pytest.fixture(scope="module", autouse=True)
def install_areal_forward():
    """Exercise AReaL's replacement with the real MindSpeed autograd function."""
    if getattr(adapter, "_NATIVE_KKT", False):
        pytest.skip("MindSpeed already provides the generalized native integration")
    original = adapter._chunk_gated_delta_rule_fwd
    try:
        assert ensure_mindspeed_gdn_native_kkt()
        assert adapter._chunk_gated_delta_rule_fwd.__module__ == (
            "areal.engine.megatron_utils.mindspeed_gdn_native"
        )
        with patch.object(
            adapter,
            "chunk_scaled_dot_kkt_fwd",
            side_effect=AssertionError("Triton KKT was called"),
        ):
            yield
    finally:
        adapter._chunk_gated_delta_rule_fwd = original


def recurrent_reference(q, k, v, g, beta, lengths):
    """Evaluate each sequence independently in FP64, including its gradients."""
    outputs = []
    start = 0
    for length in lengths:
        state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        sequence = []
        for t in range(start, start + length):
            state = state * g[:, t].exp()[..., None, None]
            residual = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
            state = (
                state
                + k[:, t, :, :, None] * (beta[:, t, :, None] * residual)[:, :, None, :]
            )
            sequence.append(
                torch.einsum("bhk,bhkv->bhv", q[:, t], state) / math.sqrt(q.shape[-1])
            )
        outputs.append(torch.stack(sequence, dim=1))
        start += length
    return torch.cat(outputs, dim=1)


def make_inputs(dtype, batch, lengths, heads, key_dim, value_dim, seed=42):
    generator = torch.Generator().manual_seed(seed)
    tokens = sum(lengths)

    def normal(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    q = torch.nn.functional.normalize(
        normal((batch, tokens, heads, key_dim)), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(
        normal((batch, tokens, heads, key_dim)), dim=-1
    ).to(dtype)
    v = normal((batch, tokens, heads, value_dim)).to(dtype)
    g = -torch.rand((batch, tokens, heads), generator=generator) * 0.2
    beta = torch.sigmoid(normal((batch, tokens, heads))).to(dtype)
    return q, k, v, g, beta


def assert_numerically_close(actual, expected, dtype, name):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert torch.isfinite(actual).all(), name
    error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
    limit = 0.02 if dtype == torch.bfloat16 else 0.006
    assert error < limit, (name, "relative_l2", error.item(), "limit", limit)
    # A norm check alone could hide an isolated large error.
    absolute = (
        0.025 if dtype == torch.bfloat16 else 0.008
    ) * expected.abs().max().item() + 1e-6
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.04 if dtype == torch.bfloat16 else 0.012,
        atol=absolute,
        msg=name,
    )


CASES = [
    # layout, batch, sequence lengths, heads, key width, value width, chunk
    ("dense", 2, (130,), 2, 128, 128, 64),
    ("packed", 1, (17, 64, 129), 2, 128, 128, 64),
    ("dense", 1, (129,), 1, 128, 128, 64),
    ("packed", 1, (1, 63, 65), 1, 128, 128, 64),
    ("dense", 2, (256,), 2, 128, 128, 128),
    ("packed", 1, (32, 96, 128), 2, 128, 128, 128),
    ("dense", 1, (65,), 2, 128, 256, 64),
    ("packed", 1, (31, 65), 2, 128, 256, 64),
    ("dense", 2, (129,), 4, 128, 256, 128),
    ("packed", 1, (17, 128, 129), 4, 128, 256, 128),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "layout,batch,lengths,heads,key_dim,value_dim,chunk_size", CASES
)
def test_native_gdn_matches_recurrence(
    dtype, layout, batch, lengths, heads, key_dim, value_dim, chunk_size
):
    """Exercise the real custom autograd function and both solve layouts."""
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    inputs = make_inputs(dtype, batch, lengths, heads, key_dim, value_dim)
    device_inputs = tuple(x.to("npu").requires_grad_() for x in inputs)
    reference_inputs = tuple(x.double().requires_grad_() for x in inputs)
    cu = (
        torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int64,
            device="npu",
        )
        if layout == "packed"
        else None
    )
    with patch.object(
        native, "npu_chunk_scaled_dot_kkt", wraps=native.npu_chunk_scaled_dot_kkt
    ) as kkt:
        with patch.object(
            adapter, "npu_solve_tri", wraps=adapter.npu_solve_tri
        ) as solve:
            actual, final_state = adapter.chunk_gated_delta_rule(
                *device_inputs,
                cu_seqlens=cu,
                chunk_size=chunk_size,
                use_qk_l2norm_in_kernel=False,
            )
    assert final_state is None
    assert kkt.call_count == 1
    assert solve.call_args.kwargs["layout"] == ("ntd" if layout == "packed" else "bnsd")
    expected = recurrent_reference(*reference_inputs, lengths)
    generator = torch.Generator().manual_seed(123)
    grad_output = torch.randn(actual.shape, generator=generator).mul_(0.1).to(dtype)
    actual_grads = torch.autograd.grad(actual, device_inputs, grad_output.to("npu"))
    reference_grads = torch.autograd.grad(
        expected, reference_inputs, grad_output.double()
    )
    assert_numerically_close(actual, expected, dtype, "output")
    for name, got, want in zip(
        ("q", "k", "v", "g", "beta"), actual_grads, reference_grads
    ):
        assert_numerically_close(got, want, dtype, name)


@pytest.mark.parametrize("chunk_size", [16, 32])
def test_native_gdn_rejects_chunks_unsupported_by_solve(chunk_size):
    """KKT's wider chunk contract must not bypass the full adapter contract."""
    inputs = make_inputs(torch.bfloat16, 1, (64,), 1, 64, 64)
    with pytest.raises(ValueError, match="chunk_size=64 or 128"):
        adapter.chunk_gated_delta_rule(*inputs, chunk_size=chunk_size)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("key_dim", [64, 256])
@pytest.mark.parametrize("packed,chunk_size", [(False, 64), (True, 128)])
def test_native_kkt_broader_key_widths_match_fp64(dtype, key_dim, packed, chunk_size):
    """KKT supports dimensions beyond those of downstream W/U kernels."""
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    lengths = (17, 64, 129) if packed else (130,)
    batch = 1 if packed else 2
    _, k, _, _, _ = make_inputs(dtype, batch, lengths, 2, key_dim, 128)
    generator = torch.Generator().manual_seed(25)
    g = -torch.rand((batch, 4, sum(lengths)), generator=generator) * 0.2
    beta = torch.rand(g.shape, generator=generator)
    k = k.transpose(1, 2).contiguous()
    cu, indices = [0], []
    for i, length in enumerate(lengths):
        cu.append(cu[-1] + length)
        for chunk in range((length + chunk_size - 1) // chunk_size):
            indices.extend((i, chunk))
    actual = (
        native.npu_chunk_scaled_dot_kkt(
            k.to("npu"),
            g.to("npu"),
            beta.to("npu"),
            cu_seqlens=cu if packed else None,
            chunk_indices=indices if packed else None,
            chunk_size=chunk_size,
        )
        .cpu()
        .double()
    )
    keys = k.double().repeat_interleave(2, dim=1)
    for begin, end in zip(cu, cu[1:]):
        for start in range(begin, end, chunk_size):
            stop = min(start + chunk_size, end)
            ks = keys[:, :, start:stop]
            gs = g[:, :, start:stop].double()
            bs = beta[:, :, start:stop].double()
            expected = (
                (ks @ ks.transpose(-1, -2))
                * (gs[..., :, None] - gs[..., None, :]).exp()
                * bs[..., :, None]
            )
            expected = expected.tril(-1)
            torch.testing.assert_close(
                actual[:, :, start:stop, : stop - start], expected, rtol=2e-4, atol=2e-5
            )
