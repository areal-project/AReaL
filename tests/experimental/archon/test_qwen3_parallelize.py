"""Tests for Qwen3 parallelization (dense and MoE).

These tests verify forward/backward passes with various parallelization strategies:
1. Dense Qwen3 with TP
2. Dense Qwen3 with TP + torch.compile
3. MoE Qwen3 with TP
4. MoE Qwen3 with EP only
5. MoE Qwen3 with TP + EP
6. MoE Qwen3 with TP + torch.compile
7. MoE Qwen3 with TP + EP + torch.compile

Run tests:
    pytest tests/experimental/archon/test_qwen3_parallelize.py -v -m multi_gpu

Note: All tests require multiple GPUs and are marked with @pytest.mark.multi_gpu.
"""

import subprocess

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# =============================================================================
# Helper Functions
# =============================================================================


def _run_qwen3_parallelize_test(n_gpus: int, test_type: str, output: str):
    """Run qwen3 parallelize test with torchrun.

    Args:
        n_gpus: Number of GPUs to use.
        test_type: Type of test to run (from TEST_REGISTRY).
        output: Output file path for test result verification.
    """
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/experimental/archon/torchrun/run_qwen3_parallelize.py",
                f"--test_type={test_type}",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")

    # Verify test result from output file
    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


# =============================================================================
# Dense Model Tests
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_dense_qwen3_tp2_forward_backward(tmp_path_factory):
    """Test dense Qwen3 with TP=2 forward/backward (2 GPUs)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "dense_tp.out"
    _run_qwen3_parallelize_test(2, "dense_tp_forward_backward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_dense_qwen3_tp2_compile_forward_backward(tmp_path_factory):
    """Test dense Qwen3 with TP=2 + torch.compile forward/backward (2 GPUs)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "dense_tp_compile.out"
    _run_qwen3_parallelize_test(2, "dense_tp_compile_forward_backward", str(output))


# =============================================================================
# MoE Model Tests
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_moe_qwen3_tp2_forward_backward(tmp_path_factory):
    """Test MoE Qwen3 with TP=2 forward/backward (2 GPUs)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "moe_tp.out"
    _run_qwen3_parallelize_test(2, "moe_tp_forward_backward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_moe_qwen3_ep2_forward_backward(tmp_path_factory):
    """Test MoE Qwen3 with EP=2 forward/backward (2 GPUs)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "moe_ep.out"
    _run_qwen3_parallelize_test(2, "moe_ep_forward_backward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_moe_qwen3_tp2_ep2_forward_backward(tmp_path_factory):
    """Test MoE Qwen3 with TP=2, EP=2 forward/backward (2 GPUs)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "moe_tp_ep.out"
    _run_qwen3_parallelize_test(2, "moe_tp_ep_forward_backward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_moe_qwen3_tp2_compile_forward_backward(tmp_path_factory):
    """Test MoE Qwen3 with TP=2 + torch.compile forward/backward (2 GPUs)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "moe_tp_compile.out"
    _run_qwen3_parallelize_test(2, "moe_tp_compile_forward_backward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_moe_qwen3_tp2_ep2_compile_forward_backward(tmp_path_factory):
    """Test MoE Qwen3 with TP=2 + EP=2 + torch.compile forward/backward (4 GPUs).

    This test covers the production scenario where TP, EP, and torch.compile
    are all enabled together (e.g., Qwen3-30B-A3B with backend=megatron:d4t2e2).

    IMPORTANT: This test requires 4 GPUs to properly test FSDP + TP + EP + compile.
    With 4 GPUs: dp_shard=2, tp=2, ep=2 → dp_shard_mod_ep=2 (enables FSDP sharding).
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs for proper FSDP + TP + EP testing")
    output = tmp_path_factory.mktemp("test_output") / "moe_tp_ep_compile.out"
    _run_qwen3_parallelize_test(4, "moe_tp_ep_compile_forward_backward", str(output))
