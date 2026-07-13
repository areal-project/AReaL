"""enable_fp8_linear() and enable_fp8_experts() unit tests.

Verifies FP8 patching, exclusion logic, alignment checks, and expert enabling.
"""

import pytest
import torch
from torch import nn

try:
    import torchao.prototype.blockwise_fp8_training.linear  # noqa: F401

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCHAO_AVAILABLE,
    reason="torchao blockwise FP8 prototype not available",
)


def _is_fp8_patched(mod: nn.Module) -> bool:
    """Check whether a module's forward has been replaced by enable_fp8_linear."""
    return (
        not hasattr(mod.forward, "__func__")
        or mod.forward.__func__ is not nn.Linear.forward
    )  # type: ignore[attr-defined]


def _is_fp8_experts_patched(mod: nn.Module) -> bool:
    """Check whether a module's forward has been replaced by enable_fp8_experts."""
    from areal.experimental.models.archon.moe.grouped_experts import GroupedExperts

    return (
        not hasattr(mod.forward, "__func__")
        or mod.forward.__func__ is not GroupedExperts.forward
    )  # type: ignore[attr-defined]


def _make_model_on_meta(**linear_kwargs):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(**linear_kwargs)

        def forward(self, x):
            return self.layer(x)

    with torch.device("meta"):
        return SimpleModel().to(torch.bfloat16)


def _make_multi_layer_model():
    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embeddings = nn.Embedding(1000, 256)
            self.q_proj = nn.Linear(256, 256, bias=False)
            self.k_proj = nn.Linear(256, 256, bias=False)
            self.w1 = nn.Linear(256, 512, bias=False)
            self.w2 = nn.Linear(512, 256, bias=False)
            self.output = nn.Linear(256, 1000, bias=False)
            self.router = nn.Linear(256, 8, bias=False)

        def forward(self, x):
            return x

    with torch.device("meta"):
        return FakeTransformer().to(torch.bfloat16)


class TestEnableFP8Linear:
    def test_basic_patching(self):
        """FP8 patching: forward replaced, weight stays BF16, class stays nn.Linear."""
        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_model_on_meta(in_features=256, out_features=512, bias=False)
        assert not _is_fp8_patched(model.layer)

        enable_fp8_linear(model)
        assert _is_fp8_patched(model.layer)
        assert model.layer.weight.dtype == torch.bfloat16
        assert type(model.layer) is nn.Linear

    def test_default_exclusions_and_patches(self):
        """Default config excludes output/router, patches eligible linears."""
        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_multi_layer_model()
        enable_fp8_linear(model)

        # Excluded by default
        assert not _is_fp8_patched(model.output)
        assert not _is_fp8_patched(model.router)
        # Eligible modules patched
        assert _is_fp8_patched(model.q_proj)
        assert _is_fp8_patched(model.k_proj)
        assert _is_fp8_patched(model.w1)
        assert _is_fp8_patched(model.w2)

    def test_skips_non_aligned_dimensions(self):
        from unittest.mock import patch

        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_model_on_meta(in_features=100, out_features=200, bias=False)
        with patch("areal.experimental.models.archon.fp8.logger") as mock_logger:
            enable_fp8_linear(model)

        assert not _is_fp8_patched(model.layer)
        warning_msgs = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("not 128-aligned" in msg for msg in warning_msgs)

    def test_skips_linear_with_bias(self):
        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_model_on_meta(in_features=256, out_features=256, bias=True)
        enable_fp8_linear(model)
        assert not _is_fp8_patched(model.layer)

    def test_custom_exclude_fqns(self):
        from areal.experimental.models.archon.fp8 import enable_fp8_linear

        model = _make_multi_layer_model()
        enable_fp8_linear(model, exclude_fqns={"proj", "output", "router"})

        assert not _is_fp8_patched(model.q_proj)
        assert _is_fp8_patched(model.w1)
        assert _is_fp8_patched(model.w2)


def _make_moe_model_on_meta():
    from areal.experimental.models.archon.moe.grouped_experts import GroupedExperts

    class FakeMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(256, 256, bias=False)
            self.experts = GroupedExperts(
                dim=256, hidden_dim=512, num_experts=4, use_grouped_mm=False
            )
            self.small_experts = GroupedExperts(
                dim=100, hidden_dim=200, num_experts=2, use_grouped_mm=False
            )

        def forward(self, x):
            return x

    with torch.device("meta"):
        return FakeMoEModel().to(torch.bfloat16)


class TestEnableFP8Experts:
    def test_basic_expert_enabling(self):
        from areal.experimental.models.archon.fp8 import enable_fp8_experts

        model = _make_moe_model_on_meta()
        enable_fp8_experts(model)

        assert _is_fp8_experts_patched(model.experts)

    def test_skips_non_aligned_experts(self):
        from areal.experimental.models.archon.fp8 import enable_fp8_experts

        model = _make_moe_model_on_meta()
        enable_fp8_experts(model)

        assert not _is_fp8_experts_patched(model.small_experts)
