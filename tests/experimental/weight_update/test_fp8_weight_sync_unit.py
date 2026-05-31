# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FP8 weight synchronization logic.

Tests generator yield order, bucket assembly, and ParamSpec generation
without requiring GPU or distributed environment.
"""

from __future__ import annotations

import torch

from areal.api import ParamSpec
from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import SchedulingStrategy
from areal.api.io_struct import WeightUpdateMeta
from areal.utils.kernel.fp8_kernel import scaled_fp8_blockwise, should_quantize_param

# ---------------------------------------------------------------------------
# Standalone helpers matching FSDPEngine logic
# ---------------------------------------------------------------------------


def _materialize_and_maybe_quantize(params, quantization, block_size, main_rank):
    """Standalone version of the generator for unit testing."""
    for name, tensor in params:
        if not main_rank:
            continue
        if quantization == "fp8" and tensor.dim() == 2 and should_quantize_param(name):
            fp8_weight, scale = scaled_fp8_blockwise(tensor, block_size)
            yield (name, fp8_weight)
            yield (name.replace(".weight", ".weight_scale_inv"), scale)
        else:
            yield (name, tensor)


def _assemble_buckets(generator, chunk_size_mb):
    """Assemble yield items into buckets based on memory limit.

    Mirrors the bucket assembly logic in
    FSDPEngine._update_weights_from_distributed.
    """
    chunk_size = chunk_size_mb * 1024 * 1024
    buffer_size = 0
    named_tensors = []
    buckets = []

    for name, tensor in generator:
        tensor_size = tensor.numel() * tensor.element_size()
        bucket_overflow = buffer_size > 0 and tensor_size + buffer_size > chunk_size
        if bucket_overflow:
            buckets.append(named_tensors)
            named_tensors = []
            buffer_size = 0
        buffer_size += tensor_size
        named_tensors.append((name, tensor))

    if named_tensors:
        buckets.append(named_tensors)

    return buckets


def _build_param_specs(named_tensors):
    """Build ParamSpec list from named tensors.

    Mirrors the logic in _update_bucket_weights_from_distributed_async.
    """
    return [
        ParamSpec(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype).split("torch.")[1],
        )
        for name, tensor in named_tensors
    ]


# ---------------------------------------------------------------------------
# Test generator yield behavior
# ---------------------------------------------------------------------------


class TestMaterializeAndMaybeQuantize:
    """Tests for _materialize_and_maybe_quantize generator."""

    def test_yield_order_weight_then_scale(self):
        """Eligible 2D weight yields (weight, scale) in that order."""
        params = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                torch.randn(256, 512, dtype=torch.bfloat16),
            ),
        ]
        result = list(
            _materialize_and_maybe_quantize(params, "fp8", [128, 128], main_rank=True)
        )

        assert len(result) == 2
        assert result[0][0] == "model.layers.0.self_attn.q_proj.weight"
        assert result[0][1].dtype == torch.float8_e4m3fn
        assert result[1][0] == "model.layers.0.self_attn.q_proj.weight_scale_inv"
        assert result[1][1].dtype == torch.float32

    def test_skip_non_2d_tensor(self):
        """1D tensors (bias, norm) are not quantized."""
        params = [
            (
                "model.layers.0.input_layernorm.weight",
                torch.randn(256, dtype=torch.bfloat16),
            ),
        ]
        result = list(
            _materialize_and_maybe_quantize(params, "fp8", [128, 128], main_rank=True)
        )

        assert len(result) == 1
        assert result[0][1].dtype == torch.bfloat16

    def test_skip_embedding(self):
        """Embedding weights are not quantized."""
        params = [
            ("model.embed_tokens.weight", torch.randn(1000, 256, dtype=torch.bfloat16)),
        ]
        result = list(
            _materialize_and_maybe_quantize(params, "fp8", [128, 128], main_rank=True)
        )

        assert len(result) == 1
        assert result[0][1].dtype == torch.bfloat16

    def test_non_main_rank_yields_nothing(self):
        """Non-main ranks do not yield tensors."""
        params = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                torch.randn(256, 512, dtype=torch.bfloat16),
            ),
        ]
        result = list(
            _materialize_and_maybe_quantize(params, "fp8", [128, 128], main_rank=False)
        )

        assert len(result) == 0

    def test_no_quantization_passthrough(self):
        """When quantization is None, all params pass through unchanged."""
        params = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                torch.randn(256, 512, dtype=torch.bfloat16),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                torch.randn(256, dtype=torch.bfloat16),
            ),
        ]
        result = list(
            _materialize_and_maybe_quantize(params, None, [128, 128], main_rank=True)
        )

        assert len(result) == 2
        assert result[0][1].dtype == torch.bfloat16
        assert result[1][1].dtype == torch.bfloat16

    def test_mixed_quantizable_and_non_quantizable(self):
        """Mixed params: some quantized, some pass-through."""
        params = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                torch.randn(256, 512, dtype=torch.bfloat16),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                torch.randn(256, dtype=torch.bfloat16),
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                torch.randn(256, 512, dtype=torch.bfloat16),
            ),
        ]
        result = list(
            _materialize_and_maybe_quantize(params, "fp8", [128, 128], main_rank=True)
        )

        assert len(result) == 5  # 2 weights + 2 scales + 1 norm
        names = [r[0] for r in result]
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.layers.0.self_attn.q_proj.weight_scale_inv" in names
        assert "model.layers.0.mlp.gate_proj.weight" in names
        assert "model.layers.0.mlp.gate_proj.weight_scale_inv" in names
        assert "model.layers.0.input_layernorm.weight" in names


# ---------------------------------------------------------------------------
# Test bucket assembly
# ---------------------------------------------------------------------------


class TestBucketAssembly:
    """Tests for _assemble_buckets memory-chunked grouping."""

    def test_single_bucket(self):
        """Small tensors all fit in one bucket."""
        params = [
            ("w1", torch.randn(256, 512, dtype=torch.bfloat16)),
            ("s1", torch.randn(2, 4, dtype=torch.float32)),
            ("w2", torch.randn(256, 512, dtype=torch.bfloat16)),
            ("s2", torch.randn(2, 4, dtype=torch.float32)),
        ]
        buckets = _assemble_buckets(iter(params), chunk_size_mb=100)

        assert len(buckets) == 1
        assert len(buckets[0]) == 4

    def test_weight_scale_paired_in_same_bucket(self):
        """Weight and its scale naturally fit in same bucket."""
        params = [
            ("layers.0.q_proj.weight", torch.randn(256, 512).to(torch.float8_e4m3fn)),
            (
                "layers.0.q_proj.weight_scale_inv",
                torch.randn(2, 4, dtype=torch.float32),
            ),
        ]
        buckets = _assemble_buckets(iter(params), chunk_size_mb=100)

        assert len(buckets) == 1
        names = [n for n, _ in buckets[0]]
        assert "layers.0.q_proj.weight" in names
        assert "layers.0.q_proj.weight_scale_inv" in names

    def test_weight_scale_split_when_weight_oversized(self):
        """When single weight exceeds bucket, scale goes to next bucket."""
        large_weight = torch.randn(4096, 4096, dtype=torch.bfloat16)  # ~32 MB
        params = [
            ("large.weight", large_weight),
            ("large.weight_scale_inv", torch.randn(32, 32, dtype=torch.float32)),
        ]
        buckets = _assemble_buckets(iter(params), chunk_size_mb=10)

        assert len(buckets) == 2
        assert buckets[0][0][0] == "large.weight"
        assert buckets[1][0][0] == "large.weight_scale_inv"

    def test_multiple_buckets(self):
        """Many tensors split across multiple buckets."""
        params = []
        for i in range(10):
            params.append((f"w{i}", torch.randn(1024, 1024, dtype=torch.bfloat16)))
        buckets = _assemble_buckets(iter(params), chunk_size_mb=1)

        assert len(buckets) > 1
        total_tensors = sum(len(b) for b in buckets)
        assert total_tensors == 10


# ---------------------------------------------------------------------------
# Test ParamSpec generation
# ---------------------------------------------------------------------------


class TestParamSpecGeneration:
    """Tests for ParamSpec list generation from named tensors."""

    def test_fp8_weight_dtype(self):
        """FP8 weight ParamSpec has float8_e4m3fn dtype."""
        named_tensors = [
            ("q_proj.weight", torch.randn(256, 512).to(torch.float8_e4m3fn)),
        ]
        specs = _build_param_specs(named_tensors)

        assert len(specs) == 1
        assert specs[0].name == "q_proj.weight"
        assert specs[0].dtype == "float8_e4m3fn"
        assert specs[0].shape == (256, 512)

    def test_scale_dtype_float32(self):
        """Scale ParamSpec has float32 dtype."""
        named_tensors = [
            ("q_proj.weight_scale_inv", torch.randn(2, 4, dtype=torch.float32)),
        ]
        specs = _build_param_specs(named_tensors)

        assert len(specs) == 1
        assert specs[0].dtype == "float32"

    def test_bf16_weight_dtype(self):
        """Non-quantized weight ParamSpec has bfloat16 dtype."""
        named_tensors = [
            ("norm.weight", torch.randn(256, dtype=torch.bfloat16)),
        ]
        specs = _build_param_specs(named_tensors)

        assert specs[0].dtype == "bfloat16"

    def test_mixed_specs(self):
        """Mixed FP8 + BF16 tensors produce correct spec list."""
        named_tensors = [
            ("q_proj.weight", torch.randn(256, 512).to(torch.float8_e4m3fn)),
            ("q_proj.weight_scale_inv", torch.randn(2, 4, dtype=torch.float32)),
            ("norm.weight", torch.randn(256, dtype=torch.bfloat16)),
        ]
        specs = _build_param_specs(named_tensors)

        assert len(specs) == 3
        dtypes = [s.dtype for s in specs]
        assert "float8_e4m3fn" in dtypes
        assert "float32" in dtypes
        assert "bfloat16" in dtypes


# ---------------------------------------------------------------------------
# Test WeightUpdateMeta serialization
# ---------------------------------------------------------------------------


class TestWeightUpdateMetaSerialization:
    """Tests for WeightUpdateMeta with quantization fields."""

    def test_from_fsdp_xccl_with_quantization(self):
        """from_fsdp_xccl preserves quantization fields."""
        from areal.api import ModelAllocation

        alloc = ModelAllocation(
            backend="fsdp",
            name="test",
            parallel=ParallelStrategy(),
            scheduling_strategy=SchedulingStrategy(),
        )
        meta = WeightUpdateMeta.from_fsdp_xccl(
            gen_allocation=alloc,
            quantization="fp8",
            quantization_config={"weight_block_size": [128, 128]},
        )

        assert meta.quantization == "fp8"
        assert meta.quantization_config == {"weight_block_size": [128, 128]}

    def test_from_megatron_xccl_with_quantization(self):
        """from_megatron_xccl preserves quantization fields."""
        from areal.api import ModelAllocation

        alloc = ModelAllocation(
            backend="megatron",
            name="test",
            parallel=ParallelStrategy(),
            scheduling_strategy=SchedulingStrategy(),
        )
        meta = WeightUpdateMeta.from_megatron_xccl(
            gen_allocation=alloc,
            quantization="fp8",
        )

        assert meta.quantization == "fp8"

    def test_with_version_preserves_quantization(self):
        """with_version() copy preserves quantization fields."""
        from areal.api import ModelAllocation

        alloc = ModelAllocation(
            backend="fsdp",
            name="test",
            parallel=ParallelStrategy(),
            scheduling_strategy=SchedulingStrategy(),
        )
        meta = WeightUpdateMeta.from_fsdp_xccl(
            gen_allocation=alloc,
            quantization="fp8",
            quantization_config={"weight_block_size": [128, 128]},
        )
        meta_v2 = meta.with_version(2)

        assert meta_v2.quantization == "fp8"
        assert meta_v2.quantization_config == {"weight_block_size": [128, 128]}
        assert meta_v2.version == 2

    def test_default_no_quantization(self):
        """Default WeightUpdateMeta has no quantization."""
        meta = WeightUpdateMeta(type="xccl")

        assert meta.quantization is None
        assert meta.quantization_config is None
