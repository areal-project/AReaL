"""Pytest wrapper for ulysses all_to_all_tensor tests.

Tests:
- test_ulysses_all_to_all_correctness: Verify output matches reference implementation
- test_ulysses_all_to_all_backward: Verify autograd backward pass works
- test_ulysses_all_to_all_compile: Verify torch.compile compatibility
"""

import subprocess

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(world_size: int, test_name: str):
    port = find_free_ports(1)[0]
    try:
        result = subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={world_size}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_ulysses_all_to_all.py",
                f"--test_name={test_name}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}\n{e.stdout}")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_ulysses_all_to_all_correctness(world_size):
    """Test that all_to_all_tensor produces correct results."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} GPUs")
    _run_test_with_torchrun(world_size, "correctness")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_ulysses_all_to_all_backward(world_size):
    """Test that autograd backward pass works correctly."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} GPUs")
    _run_test_with_torchrun(world_size, "backward")


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_ulysses_all_to_all_compile(world_size):
    """Test torch.compile compatibility (no graph breaks)."""
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} GPUs")
    _run_test_with_torchrun(world_size, "compile")
