import pytest
import torch


def pytest_collection_modifyitems(items):
    for item in items:
        if "/fp8/" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


def make_expert_weights(
    num_experts: int,
    dim: int,
    hidden_dim: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create MoE expert weights with Xavier-like 1/sqrt(fan_in) scale."""
    scale_in = dim**0.5
    scale_hidden = hidden_dim**0.5
    w1 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) / scale_in
    )
    w2 = (
        torch.randn(num_experts, dim, hidden_dim, device=device, dtype=dtype)
        / scale_hidden
    )
    w3 = (
        torch.randn(num_experts, hidden_dim, dim, device=device, dtype=dtype) / scale_in
    )
    return w1, w2, w3
