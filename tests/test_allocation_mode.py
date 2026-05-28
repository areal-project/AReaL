import pytest

from areal.api.alloc_mode import (
    AllocationValidationError,
    InvalidAllocationModeError,
    ModelAllocation,
    _AllocationMode,
)
from areal.api.cli_args import SchedulingStrategy, SchedulingStrategyType

# Test cases: dict with input string and expected properties
TEST_CASES = [
    # Inference-only
    {
        "id": "inf_only_modern_colon",
        "input": "sglang:d4p2t2",
        "num_allocs": 1,
        "gen_backend": "sglang",
        "gen_world": 16,
    },
    # Disaggregated (2 components, no names)
    {
        "id": "disagg_modern_simple",
        "input": "sglang:d2+fsdp:d4",
        "num_allocs": 2,
        "gen_backend": "sglang",
        "train_backend": "fsdp",
        "gen_dp": 2,
        "train_dp": 4,
    },
    {
        "id": "disagg_with_hybrid_moe",
        "input": "sglang:d4p2t2+megatron:(attn:d2p2t2c2|ffn:d2p2t2e2)",
        "num_allocs": 2,
        "gen_backend": "sglang",
        "train_backend": "megatron",
    },
    # Training backends
    {
        "id": "fsdp_explicit",
        "input": "fsdp:d4",
        "num_allocs": 1,
        "gen_backend": None,
        "train_backend": "fsdp",
        "train_dp": 4,
    },
    {
        "id": "megatron_explicit",
        "input": "megatron:d2p2t1",
        "num_allocs": 1,
        "train_backend": "megatron",
        "train_pp": 2,
    },
    {
        "id": "disagg_different_backends",
        "input": "vllm:d2t4+megatron:d2p2t1",
        "num_allocs": 2,
        "gen_backend": "vllm",
        "train_backend": "megatron",
        "gen_tp": 4,
        "train_pp": 2,
    },
    # Modern syntax with explicit backends
    {
        "id": "megatron_hybrid_moe",
        "input": "megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)",
        "num_allocs": 1,
        "train_backend": "megatron",
        "train_etp": 4,
    },
    {
        "id": "disagg_with_inf_and_hybrid",
        "input": "sglang:d4p1t2+megatron:(attn:d4p2t2c2|ffn:d2p2t4e2)",
        "num_allocs": 2,
        "gen_world": 8,
        "train_world": 32,
        "train_cp": 2,
    },
    # Named components
    {
        "id": "two_named_components",
        "input": "sglang[rollout]:d2+fsdp[actor]:d4",
        "num_allocs": 2,
        "names": ["rollout", "actor"],
        "rollout_backend": "sglang",
        "actor_backend": "fsdp",
    },
    {
        "id": "three_named_components",
        "input": "sglang[rollout]:d2+fsdp[actor]:d4+fsdp[critic]:d4",
        "num_allocs": 3,
        "names": ["rollout", "actor", "critic"],
    },
    # Colocation
    {
        "id": "colocation_with_names",
        "input": "sglang[rollout]:d2+fsdp[actor]:d4|fsdp[critic]:d4",
        "num_allocs": 3,
        "rollout_sched": SchedulingStrategyType.separation.value,
        "actor_sched": SchedulingStrategyType.separation.value,
        "critic_sched": SchedulingStrategyType.colocation.value,
        "critic_target": "actor",
    },
    {
        "id": "multi_agent_allocation",
        "input": "vllm[rollout1]:d2t2 + fsdp[actor1]:d4 + vllm[rollout2]:d4 + fsdp[actor2]:d4",
        "num_allocs": 4,
        "names": ["rollout1", "actor1", "rollout2", "actor2"],
    },
]

VALIDATION_ERROR_CASES = [
    # FSDP unsupported configurations
    {
        "id": "fsdp_pipeline",
        "input": "fsdp:d4p2",
        "error": AllocationValidationError,
        "match": "FSDP backend only supports data/tensor/context parallelism",
    },
    {
        "id": "fsdp_pipeline",
        "input": "fsdp:d2e2",
        "error": AllocationValidationError,
        "match": "FSDP backend only supports data/tensor/context parallelism",
    },
    {
        "id": "three_plus_without_names",
        "input": "sglang:d2+fsdp[actor]:d4+fsdp:d4",
        "error": AllocationValidationError,
        "match": "all must have names",
    },
    {
        "id": "duplicate_names",
        "input": "sglang[actor]:d2+fsdp[actor]:d4",
        "error": AllocationValidationError,
        "match": "Duplicate component name",
    },
    {
        "id": "colocation_without_names",
        "input": "fsdp:d4|fsdp:d4",
        "error": AllocationValidationError,
        "match": "must have names",
    },
    {
        "id": "hybrid_pp_mismatch",
        "input": "(attn:d2p2t1c4|ffn:d2p4t1e2)",
        "error": AllocationValidationError,
        "match": "Pipeline parallel size.*must be identical",
    },
    {
        "id": "hybrid_world_mismatch",
        "input": "(attn:d4p2t1c1|ffn:d2p2t2e2)",
        "error": InvalidAllocationModeError,
        "match": "World size.*must be identical",
    },
    # Bare dims without explicit backend prefix
    {
        "id": "bare_dims_simple",
        "input": "d1",
        "error": AllocationValidationError,
        "match": "must be explicitly specified",
    },
    {
        "id": "bare_dims_complex",
        "input": "d2p2t1",
        "error": AllocationValidationError,
        "match": "must be explicitly specified",
    },
    {
        "id": "bare_dims_with_name",
        "input": "[actor]:d4",
        "error": AllocationValidationError,
        "match": "must be explicitly specified",
    },
    {
        "id": "bare_dims_in_multi_component",
        "input": "sglang:d4+d4",
        "error": AllocationValidationError,
        "match": "must be explicitly specified",
    },
    {
        "id": "bare_dims_named_in_multi",
        "input": "vllm:d4+[actor]:d4",
        "error": AllocationValidationError,
        "match": "must be explicitly specified",
    },
    {
        "id": "bare_hybrid_moe_no_backend",
        "input": "attn:d4p2t2c2|ffn:d2p2t4e2",
        "error": AllocationValidationError,
        "match": "must be explicitly specified",
    },
    {
        "id": "bare_hybrid_moe_parens_no_backend",
        "input": "(attn:d4p2t2c2|ffn:d2p2t4e2)",
        "error": AllocationValidationError,
        "match": "must be explicitly specified",
    },
]


@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc["id"])
def test_allocation_parsing(test_case):
    """Test allocation mode parsing with various configurations."""
    mode = _AllocationMode.from_str(test_case["input"])

    # Check number of allocations
    assert len(mode.allocations) == test_case["num_allocs"]

    # Check gen properties (if expected)
    if "gen_dp" in test_case:
        assert mode.gen.dp_size == test_case["gen_dp"]
    if "gen_tp" in test_case:
        assert mode.gen.tp_size == test_case["gen_tp"]
    if "gen_pp" in test_case:
        assert mode.gen.pp_size == test_case["gen_pp"]
    if "gen_world" in test_case:
        assert mode.gen.world_size == test_case["gen_world"]
    if "gen_backend" in test_case:
        assert mode.gen_backend == test_case["gen_backend"]

    # Check train properties (if expected)
    if "train_dp" in test_case:
        assert mode.train.dp_size == test_case["train_dp"]
    if "train_tp" in test_case:
        assert mode.train.tp_size == test_case["train_tp"]
    if "train_pp" in test_case:
        assert mode.train.pp_size == test_case["train_pp"]
    if "train_cp" in test_case:
        assert mode.train.cp_size == test_case["train_cp"]
    if "train_ep" in test_case:
        assert mode.train.ep_size == test_case["train_ep"]
    if "train_etp" in test_case:
        assert mode.train.etp_size == test_case["train_etp"]
    if "train_edp" in test_case:
        assert mode.train.edp_size == test_case["train_edp"]
    if "train_world" in test_case:
        assert mode.train.world_size == test_case["train_world"]
    if "train_backend" in test_case:
        assert mode.train_backend == test_case["train_backend"]

    # Check named component access
    if "names" in test_case:
        for name in test_case["names"]:
            assert mode[name].name == name

    # Check specific named component properties
    for key, value in test_case.items():
        if "_backend" in key and key not in ["gen_backend", "train_backend"]:
            name = key.replace("_backend", "")
            assert mode[name].backend == value
        elif "_sched" in key:
            name = key.replace("_sched", "")
            assert mode[name].scheduling_strategy.type == value
        elif "_target" in key:
            name = key.replace("_target", "")
            assert mode[name].scheduling_strategy.target == value


@pytest.mark.parametrize("test_case", VALIDATION_ERROR_CASES, ids=lambda tc: tc["id"])
def test_validation_errors(test_case):
    """Test that validation errors are raised correctly."""
    with pytest.raises(test_case["error"], match=test_case["match"]):
        _AllocationMode.from_str(test_case["input"])


def test_backward_compatible_properties():
    """Test backward-compatible properties work correctly."""
    # Unambiguous case
    mode = _AllocationMode.from_str("sglang:d2+fsdp:d4")
    assert mode.gen.dp_size == 2
    assert mode.train.dp_size == 4
    assert mode.gen_backend == "sglang"
    assert mode.train_backend == "fsdp"
    assert mode.gen_instance_size == 1

    # Ambiguous gen property
    mode = _AllocationMode.from_str("sglang[r1]:d2+sglang[r2]:d2+fsdp[actor]:d4")
    with pytest.raises(AttributeError, match="Ambiguous"):
        _ = mode.gen

    # Ambiguous train property
    mode = _AllocationMode.from_str("sglang[rollout]:d2+fsdp[actor]:d4+fsdp[critic]:d4")
    with pytest.raises(AttributeError, match="Ambiguous"):
        _ = mode.train


def test_getitem_access():
    """Test __getitem__ access by name."""
    mode = _AllocationMode.from_str("sglang[rollout]:d2+fsdp[actor]:d4")
    rollout = mode["rollout"]
    assert isinstance(rollout, ModelAllocation)
    assert rollout.backend == "sglang"
    assert rollout.name == "rollout"

    # Non-existent name
    with pytest.raises(KeyError):
        _ = mode["nonexistent"]


def test_operator_precedence():
    """Test that | binds tighter than +."""
    # a + b|c should parse as a + (b|c), not (a+b)|c
    mode = _AllocationMode.from_str("sglang[rollout]:d2+fsdp[actor]:d4|fsdp[critic]:d4")
    assert len(mode.allocations) == 3
    # rollout is separate
    assert (
        mode["rollout"].scheduling_strategy.type
        == SchedulingStrategyType.separation.value
    )
    # actor and critic are colocated
    assert (
        mode["actor"].scheduling_strategy.type
        == SchedulingStrategyType.separation.value
    )
    assert (
        mode["critic"].scheduling_strategy.type
        == SchedulingStrategyType.colocation.value
    )
    assert mode["critic"].scheduling_strategy.target == "actor"


class TestModelAllocationFromStr:
    """Tests for ModelAllocation.from_str() single-component parser."""

    def test_fsdp_simple(self):
        """Test parsing simple FSDP allocation."""
        alloc = ModelAllocation.from_str("fsdp:d4")
        assert alloc.backend == "fsdp"
        assert alloc.parallel.data_parallel_size == 4
        assert alloc.parallel.tensor_parallel_size == 1

    def test_sglang_with_tp(self):
        """Test parsing SGLang allocation with tensor parallelism."""
        alloc = ModelAllocation.from_str("sglang:d4t2")
        assert alloc.backend == "sglang"
        assert alloc.parallel.data_parallel_size == 4
        assert alloc.parallel.tensor_parallel_size == 2

    def test_vllm_with_pp(self):
        """Test parsing vLLM allocation with pipeline parallelism."""
        alloc = ModelAllocation.from_str("vllm:d2t4p2")
        assert alloc.backend == "vllm"
        assert alloc.parallel.data_parallel_size == 2
        assert alloc.parallel.tensor_parallel_size == 4
        assert alloc.parallel.pipeline_parallel_size == 2

    def test_megatron_full(self):
        """Test parsing Megatron allocation with all dimensions."""
        alloc = ModelAllocation.from_str("megatron:d4t2p2")
        assert alloc.backend == "megatron"
        assert alloc.parallel.data_parallel_size == 4
        assert alloc.parallel.tensor_parallel_size == 2
        assert alloc.parallel.pipeline_parallel_size == 2

    def test_archon_simple(self):
        """Test parsing Archon allocation."""
        alloc = ModelAllocation.from_str("archon:d2")
        assert alloc.backend == "archon"
        assert alloc.parallel.data_parallel_size == 2

    def test_hybrid_moe(self):
        """Test parsing hybrid MoE allocation with attn/ffn split."""
        alloc = ModelAllocation.from_str("megatron:(attn:d1p12t4|ffn:d1p12e4)")
        assert alloc.backend == "megatron"
        assert alloc.parallel.tensor_parallel_size == 4
        assert alloc.parallel.pipeline_parallel_size == 12
        assert alloc.parallel.expert_parallel_size == 4

    def test_with_name(self):
        """Test overriding name parameter."""
        alloc = ModelAllocation.from_str("fsdp:d4", name="actor")
        assert alloc.name == "actor"
        assert alloc.backend == "fsdp"

    def test_with_scheduling_strategy(self):
        """Test overriding scheduling strategy parameter."""
        sched = SchedulingStrategy(
            type=SchedulingStrategyType.colocation.value, target="actor"
        )
        alloc = ModelAllocation.from_str("fsdp:d4", scheduling_strategy=sched)
        assert alloc.scheduling_strategy.type == "colocation"
        assert alloc.scheduling_strategy.target == "actor"

    def test_default_scheduling_is_separation(self):
        """Test that default scheduling strategy is separation."""
        alloc = ModelAllocation.from_str("fsdp:d4")
        assert alloc.scheduling_strategy.type == "separation"

    def test_world_size(self):
        """Test world_size computation from parallel dimensions."""
        alloc = ModelAllocation.from_str("fsdp:d4t2")
        assert alloc.parallel.world_size == 8  # 4*2*1*1

    def test_context_parallel(self):
        """Test parsing context parallelism dimension."""
        alloc = ModelAllocation.from_str("fsdp:d4c2")
        assert alloc.parallel.context_parallel_size == 2

    def test_single_gpu(self):
        """Test single-GPU allocation."""
        alloc = ModelAllocation.from_str("fsdp:d1")
        assert alloc.parallel.world_size == 1

    def test_sglang_d1(self):
        """Test single-GPU SGLang allocation."""
        alloc = ModelAllocation.from_str("sglang:d1")
        assert alloc.backend == "sglang"
        assert alloc.parallel.world_size == 1

    def test_rejects_multi_component(self):
        """Test that from_str rejects multi-component strings containing '+'."""
        with pytest.raises(ValueError, match="Multi-component strings containing"):
            ModelAllocation.from_str("sglang:d4+fsdp:d4")

    def test_rejects_multi_component_three_parts(self):
        """Test rejection of three-component strings."""
        with pytest.raises(ValueError, match="Multi-component strings containing"):
            ModelAllocation.from_str("sglang:d2+fsdp:d4+megatron:d8")


class TestTrainEngineConfigBackendNormalization:
    """Tests for TrainEngineConfig.__post_init__ backend normalization."""

    def test_explicit_fsdp_preserved(self):
        """Test that explicit fsdp backend is preserved."""
        from areal.api.cli_args import TrainEngineConfig

        config = TrainEngineConfig(backend="fsdp:d4")
        assert config.backend == "fsdp:d4"

    def test_explicit_megatron_preserved(self):
        """Test that explicit megatron backend is preserved."""
        from areal.api.cli_args import TrainEngineConfig

        config = TrainEngineConfig(backend="megatron:d4t2p2")
        assert config.backend == "megatron:d4t2p2"

    def test_explicit_archon_preserved(self):
        """Test that explicit archon backend is preserved."""
        from areal.api.cli_args import TrainEngineConfig

        config = TrainEngineConfig(backend="archon:d2")
        assert config.backend == "archon:d2"

    def test_hybrid_moe_preserved(self):
        """Test that hybrid MoE backend spec is preserved without crashing."""
        from areal.api.cli_args import TrainEngineConfig

        spec = "megatron:(attn:d1p12t4|ffn:d1p12e4)"
        config = TrainEngineConfig(backend=spec)
        assert "megatron" in config.backend

    def test_single_gpu_default(self):
        """Test single-GPU fsdp:d1 backend is correctly normalized."""
        from areal.api.cli_args import TrainEngineConfig

        config = TrainEngineConfig(backend="fsdp:d1")
        assert config.backend == "fsdp:d1"


class TestInferenceEngineConfigBackendNormalization:
    """Tests for InferenceEngineConfig.__post_init__ backend normalization."""

    def test_explicit_sglang_preserved(self):
        """Test that explicit sglang backend is preserved."""
        from areal.api.cli_args import InferenceEngineConfig

        config = InferenceEngineConfig(backend="sglang:d4t2")
        assert config.backend == "sglang:d4t2"

    def test_explicit_vllm_preserved(self):
        """Test that explicit vllm backend is preserved."""
        from areal.api.cli_args import InferenceEngineConfig

        config = InferenceEngineConfig(backend="vllm:d2t4")
        assert config.backend == "vllm:d2t4"

    def test_single_gpu_default(self):
        """Test single-GPU sglang:d1 backend is correctly normalized."""
        from areal.api.cli_args import InferenceEngineConfig

        config = InferenceEngineConfig(backend="sglang:d1")
        assert config.backend == "sglang:d1"
