# SPDX-License-Identifier: Apache-2.0
"""Tests for :class:`AwexFSDPAdapter` dtype invariants.

Regression guard for the awex-side counterpart of PR #1369's fp32 master
weight fix. FSDPEngine stores parameters in ``optimizer_dtype`` (fp32 by
default) but rollout engines expect ``compute_dtype`` (bf16). The awex
adapter must cast on send and report ``compute_dtype`` in metadata,
otherwise NCCL sees a byte-count mismatch (train fp32 = 4B/elem vs
rollout bf16 = 2B/elem) and hangs at first transfer.

These are CPU-only tests: they exercise the metadata / cast wiring on a
plain ``nn.Module`` with fp32 parameters and assert the adapter reports
and returns bf16. A full NCCL round-trip requires multi-GPU and is
covered by the ``torchrun/`` integration tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# awex depends on megatron at import time; skip cleanly in dev envs that
# do not ship megatron.
pytest.importorskip("awex")
pytest.importorskip("awex.meta.weight_meta")

from areal.v2.weight_update.awex.fsdp_adapter import AwexFSDPAdapter  # noqa: E402


class _TinyModel(nn.Module):
    """Two fp32 parameters, one with a non-trivial name for HF renaming."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4, 2, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))


def _make_engine(compute_dtype: torch.dtype) -> MagicMock:
    """Mock FSDPEngine surface used by ``AwexFSDPAdapter``.

    Only the attributes the adapter touches on the plain-tensor (non-DTensor)
    code path are stubbed. That covers the metadata + cast invariants without
    pulling in FSDP2 / DeviceMesh setup.
    """
    engine = MagicMock()
    engine.model = _TinyModel()

    # Non-vision model → _to_hf_name is a no-op passthrough.
    engine.is_vision_model = False
    engine.model_config.model_type = "llama"
    engine.model_config.tie_word_embeddings = False

    # RankInfo inputs — single-rank, no parallelism.
    mesh = MagicMock()
    mesh.mesh_dim_names = ()
    mesh.size.return_value = 1
    mesh.get_local_rank.return_value = 0
    engine.world_mesh = mesh
    engine.world_size = 1
    engine.rank = 0
    engine.data_parallel_world_size = 1
    engine.dp_rank = 0

    # This is the core of the regression: adapter must call these two.
    engine._compute_dtype.return_value = compute_dtype

    def _cast(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point() and t.dtype != compute_dtype:
            return t.to(compute_dtype)
        return t

    engine._cast_to_compute_dtype.side_effect = _cast
    return engine


def test_get_weight_metadata_reports_compute_dtype_not_storage_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ParameterMeta.dtype and ParameterShardMeta.dtype must both be the
    compute dtype the rollout side expects, even when the underlying
    tensors are stored in fp32 master precision (PR #1369)."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    engine = _make_engine(compute_dtype=torch.bfloat16)
    adapter = AwexFSDPAdapter(engine)

    metadata = adapter.get_weight_metadata()

    assert len(metadata) == 2, "expected one ParameterMeta per parameter"
    for meta in metadata:
        assert meta.dtype == torch.bfloat16, (
            f"ParameterMeta.dtype for {meta.name} is {meta.dtype}; "
            f"must be compute dtype (bf16), not fp32 storage dtype"
        )
        assert len(meta.shards) == 1
        assert meta.shards[0].dtype == torch.bfloat16, (
            f"ParameterShardMeta.dtype for {meta.name} is {meta.shards[0].dtype}; "
            f"must be compute dtype (bf16), not fp32 storage dtype"
        )


def test_get_local_shard_parameters_returns_compute_dtype_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tensors handed to NCCL must be in compute dtype so the wire
    byte count matches ParameterMeta.dtype and the rollout receive
    buffer."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    engine = _make_engine(compute_dtype=torch.bfloat16)
    adapter = AwexFSDPAdapter(engine)

    params = adapter.get_local_shard_parameters()

    assert set(params.keys()) == {"weight", "bias"}
    for name, tensor in params.items():
        assert tensor.dtype == torch.bfloat16, (
            f"{name} handed to NCCL is {tensor.dtype}; must be compute "
            f"dtype (bf16) to match the sender-side ParameterMeta.dtype"
        )


def test_no_op_when_storage_already_matches_compute_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When storage dtype already equals compute dtype (e.g. user set
    optimizer_dtype=bf16 explicitly), the adapter must still report
    that dtype and hand through tensors unchanged."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    # Model tensors are fp32 by default; force compute dtype to fp32 so
    # cast is a no-op path.
    engine = _make_engine(compute_dtype=torch.float32)
    adapter = AwexFSDPAdapter(engine)

    metadata = adapter.get_weight_metadata()
    params = adapter.get_local_shard_parameters()

    for meta in metadata:
        assert meta.dtype == torch.float32
        assert meta.shards[0].dtype == torch.float32
    for tensor in params.values():
        assert tensor.dtype == torch.float32
