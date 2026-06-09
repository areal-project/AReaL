"""Unit tests for PD disaggregation allocation parsing.

Run:
    uv run pytest tests/test_pd_alloc_mode.py -v
"""

from areal.api.alloc_mode import ModelAllocation, _LLMParallelParser


class TestPDSyntaxParsing:
    """Verify that the Lark parser handles sglang(P:...|D:...) syntax."""

    def test_basic_pd_spec(self):
        """P:d1t1p1|D:d1t1p1 should produce two groups with dp=1 each."""
        allocs = ModelAllocation.from_str_multi("sglang(P:d1t1p1|D:d1t1p1)")
        assert len(allocs) == 2
        assert allocs[0].name == "P"
        assert allocs[0].parallel.dp_size == 1
        assert allocs[0].parallel.tensor_parallel_size == 1
        assert allocs[1].name == "D"
        assert allocs[1].parallel.dp_size == 1

    def test_asymmetric_pd_spec(self):
        """Different TP sizes across P and D groups."""
        allocs = ModelAllocation.from_str_multi("sglang(P:d2t2p1|D:d2t1p1)")
        assert len(allocs) == 2
        assert allocs[0].parallel.tensor_parallel_size == 2
        assert allocs[1].parallel.tensor_parallel_size == 1

    def test_from_str_returns_synthetic_allocation(self):
        """from_str merges PD groups into a single synthetic allocation."""
        alloc = ModelAllocation.from_str("sglang(P:d1t1p1|D:d1t1p1)")
        assert alloc.backend == "sglang"
        assert alloc.parallel.dp_size == 2  # sum of P + D
        assert hasattr(alloc, "_pd_groups")
        assert len(alloc._pd_groups) == 2

    def test_from_str_single_allocation_unchanged(self):
        """Non-PD specs should work exactly as before."""
        alloc = ModelAllocation.from_str("sglang:d2t1")
        assert alloc.backend == "sglang"
        assert alloc.parallel.dp_size == 2
        assert not hasattr(alloc, "_pd_groups")

    def test_parser_raw_output(self):
        """The Lark parser returns a list for PD specs."""
        parser = _LLMParallelParser()
        result = parser.parse("sglang(P:d1t1p1|D:d1t1p1)")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_pd_groups_inherit_backend(self):
        """All PD groups should inherit the outer backend name."""
        allocs = ModelAllocation.from_str_multi("sglang(P:d1t1p1|D:d1t1p1)")
        for a in allocs:
            assert a.backend == "sglang"

    def test_pd_groups_use_separation_strategy(self):
        """PD groups should all use separation scheduling."""
        from areal.api.alloc_mode import SchedulingStrategyType

        allocs = ModelAllocation.from_str_multi("sglang(P:d1t1p1|D:d1t1p1)")
        for a in allocs:
            assert a.scheduling_strategy.type == SchedulingStrategyType.separation

    def test_allocation_mode_pd_spec(self):
        """_AllocationMode.from_str should handle PD specs."""
        from areal.api.alloc_mode import _AllocationMode

        mode = _AllocationMode.from_str("sglang(P:d1t1p1|D:d1t1p1)")
        assert len(mode.allocations) == 2

    def test_from_str_multi_non_pd_returns_single(self):
        """from_str_multi on a non-PD spec should return a list of one."""
        allocs = ModelAllocation.from_str_multi("sglang:d2t1")
        assert len(allocs) == 1
        assert allocs[0].parallel.dp_size == 2

    def test_pd_spec_with_pipeline(self):
        """PD groups can include pipeline parallelism."""
        allocs = ModelAllocation.from_str_multi("sglang(P:d1t1p2|D:d1t1p1)")
        assert allocs[0].parallel.pipeline_parallel_size == 2
        assert allocs[1].parallel.pipeline_parallel_size == 1
