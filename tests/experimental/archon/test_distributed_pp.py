"""Pipeline Parallelism (PP) distributed tests for Archon Engine.

These tests require multiple GPUs and use torchrun for distributed execution.

Run tests:
    pytest tests/experimental/archon/test_distributed_pp.py -v -m multi_gpu

Test configuration:
    2 GPU Tests (Core PP - manual P2P):
        - test_pp_forward_2gpu: PP=2, manual activation passing (1F1B)
        - test_pp_backward_2gpu: PP=2, manual gradient passing (1F1B)
        - test_pp_gradient_correctness_2gpu: PP=2, tests PP gradients match non-PP

    4 GPU Tests (Extended PP - manual P2P):
        - test_pp_forward_4gpu: PP=4, manual activation passing (1F1B)
        - test_pp_backward_4gpu: PP=4, manual gradient passing (1F1B)

    Schedule API Tests (2 GPU):
        - test_pp_zbv_2gpu[forward_schedule]: PP=2, schedule.eval() with ZBVZeroBubble
        - test_pp_zbv_2gpu[backward_schedule]: PP=2, schedule.step() with ZBVZeroBubble

    Schedule API Tests (4 GPU):
        - test_pp_izb_4gpu[forward_schedule]: PP=4, schedule.eval() with InterleavedZeroBubble
        - test_pp_izb_4gpu[backward_schedule]: PP=4, schedule.step() with InterleavedZeroBubble

    PP Combination Tests (4 GPU):
        - test_pp_tp_forward_4gpu: PP=2, TP=2, tests PP+TP combination
        - test_pp_dp_forward_4gpu: PP=2, DP=2, tests PP+DP combination
        - test_pp_ep_forward_4gpu: PP=2, EP=2, tests PP+EP combination (MoE model)

    PP Checkpoint Tests (2 GPU):
        - test_pp_dcp_checkpoint_2gpu: PP=2, tests DCP save/load
        - test_pp_dcp_with_optim_2gpu: PP=2, tests DCP with optimizer state
        - test_pp_forward_match_2gpu: PP=2, tests forward match after checkpoint
"""

import subprocess
import tempfile

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_pp_test_with_torchrun(
    script: str,
    n_gpus: int,
    extra_args: list[str] | None = None,
    timeout: int = 300,
):
    """Run a PP test script with torchrun.

    Args:
        script: Path to the test script
        n_gpus: Number of GPUs to use
        extra_args: Additional command line arguments
        timeout: Timeout in seconds

    Raises:
        pytest.fail: If the test fails
    """
    port = find_free_ports(1)[0]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        out_file = f.name

    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        script,
        f"--output={out_file}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Check result file
        with open(out_file) as f:
            test_result = f.read().strip()

        if test_result != "Passed":
            pytest.fail(
                f"Test returned '{test_result}'. "
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}\nstdout: {e.stdout}")
    except subprocess.TimeoutExpired:
        pytest.fail(f"Test timed out after {timeout} seconds")


# =============================================================================
# 2 GPU Tests (Core PP tests)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_forward_2gpu():
    """Test PP forward pass with 2 GPUs (pp=2) via manual P2P.

    Validates that PP model output matches golden (non-PP) model output
    using manual activation passing between stages (1F1B only).
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_tests.py",
        n_gpus=2,
        extra_args=["--test_type=forward_p2p", "--pp_size=2"],
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_backward_2gpu():
    """Test PP backward pass with 2 GPUs (pp=2) via manual P2P.

    Validates that gradients flow correctly through all PP stages
    using manual gradient passing between stages (1F1B only).
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_tests.py",
        n_gpus=2,
        extra_args=["--test_type=backward_p2p", "--pp_size=2"],
    )


@pytest.mark.multi_gpu
def test_pp_gradient_correctness_2gpu():
    """Test PP gradient correctness with 2 GPUs (pp=2).

    Validates that PP step() API produces identical gradients to
    manual forward/backward passes without pipeline parallelism.

    This test uses a simple model with:
    - FirstStageModel: embedding + 2 transformer blocks
    - LastStageModel: 2 transformer blocks + output head
    - 2 microbatches with different packed sequence lengths
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_gradient_verify.py",
        n_gpus=2,
        extra_args=["--n_microbatches=2", "--seq_len=64"],
        timeout=120,
    )


# =============================================================================
# Schedule API Tests (2 GPU)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("test_type", ["forward_schedule", "backward_schedule"])
def test_pp_zbv_2gpu(test_type: str):
    """Test ZBVZeroBubble forward/backward pass with 2 GPUs (pp=2) via schedule API.

    Validates that PP model with ZBVZeroBubble schedule produces correct output
    using schedule.eval() and that gradients flow correctly using schedule.step()
    with V-style stage assignment.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_tests.py",
        n_gpus=2,
        extra_args=[
            f"--test_type={test_type}",
            "--pp_size=2",
            "--pp_schedule=ZBVZeroBubble",
        ],
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("test_type", ["forward_schedule", "backward_schedule"])
def test_pp_izb_4gpu(test_type: str):
    """Test InterleavedZeroBubble forward/backward pass with 4 GPUs (pp=4) via schedule API.

    Validates that PP model with InterleavedZeroBubble (ZB1P) schedule produces
    correct output using schedule.eval() and that gradients flow correctly using
    schedule.step() with loop-style stage assignment.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_tests.py",
        n_gpus=4,
        extra_args=[
            f"--test_type={test_type}",
            "--pp_size=4",
            "--pp_schedule=InterleavedZeroBubble",
        ],
    )


# =============================================================================
# 4 GPU Tests (Extended PP tests)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_forward_4gpu():
    """Test PP forward pass with 4 GPUs (pp=4) via manual P2P.

    Validates PP with more stages (4 stages instead of 2) using
    manual activation passing (1F1B only).
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_tests.py",
        n_gpus=4,
        extra_args=["--test_type=forward_p2p", "--pp_size=4"],
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_backward_4gpu():
    """Test PP backward pass with 4 GPUs (pp=4) via manual P2P.

    Validates gradient flow with more stages using manual gradient
    passing (1F1B only).
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_tests.py",
        n_gpus=4,
        extra_args=["--test_type=backward_p2p", "--pp_size=4"],
    )


# =============================================================================
# PP Combination Tests (PP+TP, PP+DP, PP+EP)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_tp_forward_4gpu():
    """Test PP+TP combination with 4 GPUs (pp=2, tp=2).

    Validates that PP works correctly when combined with Tensor Parallelism.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_combinations.py",
        n_gpus=4,
        extra_args=["--test_type=pp_tp"],
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_dp_forward_4gpu():
    """Test PP+DP combination with 4 GPUs (pp=2, dp_shard=2).

    Validates that PP works correctly when combined with Data Parallelism.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_combinations.py",
        n_gpus=4,
        extra_args=["--test_type=pp_dp"],
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_ep_forward_4gpu():
    """Test PP+EP combination with 4 GPUs (pp=2, ep=2).

    Validates that PP works correctly when combined with Expert Parallelism.
    This test uses a MoE model since EP requires expert parallelism.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_pp_combinations.py",
        n_gpus=4,
        extra_args=["--test_type=pp_ep"],
    )


# =============================================================================
# PP Checkpoint Tests
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_dcp_checkpoint_2gpu():
    """Test PP checkpoint save/load using DCP format with 2 GPUs.

    Validates that PP model weights can be saved and loaded correctly.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_checkpoint_tests.py",
        n_gpus=2,
        extra_args=["--test_type=pp_dcp_checkpoint", "--pp_size=2"],
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_dcp_with_optim_2gpu():
    """Test PP checkpoint with optimizer state using DCP format with 2 GPUs.

    Validates that optimizer state is correctly saved and loaded with PP.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_checkpoint_tests.py",
        n_gpus=2,
        extra_args=["--test_type=pp_dcp_with_optim", "--pp_size=2"],
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_pp_forward_match_2gpu():
    """Test forward output matches after PP checkpoint save/load with 2 GPUs.

    Validates that model behavior is preserved after checkpoint save/load.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    _run_pp_test_with_torchrun(
        "tests/experimental/archon/torchrun/run_checkpoint_tests.py",
        n_gpus=2,
        extra_args=["--test_type=pp_forward_match", "--pp_size=2"],
    )
