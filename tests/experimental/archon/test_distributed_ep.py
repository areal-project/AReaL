"""Expert Parallelism (EP) and Expert Tensor Parallelism (ETP) tests for Archon Engine.

Tests various MoE parallelism configurations:
- EP: Expert Parallel where experts are sharded across ranks
- ETP: Expert Tensor Parallel with 2D weight sharding
- EP+TP: Expert Parallel with Tensor Parallel
- EP+CP: Expert Parallel with Context Parallel
- TP-only MoE: Tensor Parallel for MoE when EP is disabled

Run tests:
    pytest tests/experimental/archon/test_distributed_ep.py -v -m multi_gpu
"""

import subprocess

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_ep_test_with_torchrun(n_gpus: int, test_type: str, output: str):
    """Run EP/ETP test with torchrun.

    Args:
        n_gpus: Number of GPUs to use
        test_type: Type of test from run_ep_tests.py
        output: Output file path for test result verification
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
                "tests/experimental/archon/torchrun/run_ep_tests.py",
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
# EP + TP Tests (ep=world_size, tp=world_size, etp=1)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_tp_forward_2gpu(tmp_path_factory):
    """Test EP+TP forward numerical correctness (ep=2, tp=2, etp=1) on 2 GPUs.

    Verify: EP model output matches non-EP golden model output.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_tp_forward.out"
    _run_ep_test_with_torchrun(2, "ep_tp_forward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_tp_weight_sync_2gpu(tmp_path_factory):
    """Test EP+TP weight gather and roundtrip (ep=2, tp=2, etp=1) on 2 GPUs.

    Verify: Gathered weights match original, cross-rank consistency, roundtrip.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_tp_weight_sync.out"
    _run_ep_test_with_torchrun(2, "ep_tp_weight_sync", str(output))


# =============================================================================
# EP Only Tests (ep=world_size, tp=1, cp=1)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_only_forward_2gpu(tmp_path_factory):
    """Test EP only forward (ep=2, tp=1, cp=1) on 2 GPUs.

    Verify: EP model output matches non-EP golden model output.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_only_forward.out"
    _run_ep_test_with_torchrun(2, "ep_only_forward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_only_weight_sync_2gpu(tmp_path_factory):
    """Test EP only weight sync (ep=2, tp=1, cp=1) on 2 GPUs.

    Verify: Gathered weights match original, cross-rank consistency, roundtrip.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_only_weight_sync.out"
    _run_ep_test_with_torchrun(2, "ep_only_weight_sync", str(output))


# =============================================================================
# EP + CP Tests (ep=4, tp=1, cp=2, 4 GPU)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_cp_forward_4gpu(tmp_path_factory):
    """Test EP+CP forward (ep=4, tp=1, cp=2) on 4 GPUs.

    Verify: EP+CP model output matches non-EP golden model output.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_cp_forward.out"
    _run_ep_test_with_torchrun(4, "ep_cp_forward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_cp_weight_sync_4gpu(tmp_path_factory):
    """Test EP+CP weight sync (ep=4, tp=1, cp=2) on 4 GPUs.

    Verify: Gathered weights match original, cross-rank consistency, roundtrip.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_cp_weight_sync.out"
    _run_ep_test_with_torchrun(4, "ep_cp_weight_sync", str(output))


# =============================================================================
# ETP Forward/Weight Sync Tests (ep>1, etp=tp) - Requires 4 GPUs
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_etp_forward_4gpu(tmp_path_factory):
    """Test ETP forward numerical correctness (ep=2, tp=2, etp=2) on 4 GPUs.

    Tests ExpertTensorParallel with 2D weight sharding [Shard(0), Shard(1/2)].
    Verify: ETP model output matches non-EP golden model output.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "etp_forward.out"
    _run_ep_test_with_torchrun(4, "etp_forward", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_etp_weight_sync_4gpu(tmp_path_factory):
    """Test ETP weight gather and roundtrip (ep=2, tp=2, etp=2) on 4 GPUs.

    Tests weight gather/roundtrip with ExpertTensorParallel.
    Verify: Gathered weights match original, cross-rank consistency, roundtrip.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "etp_weight_sync.out"
    _run_ep_test_with_torchrun(4, "etp_weight_sync", str(output))


# =============================================================================
# TP-Only MoE Tests (ep=1, tp>1)
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_tp_only_moe_forward_2gpu(tmp_path_factory):
    """Test TP-only forward for MoE experts (ep=1, tp=2) on 2 GPUs.

    Tests TensorParallel class for MoE experts when EP is disabled.
    Verify: TP-only MoE model output matches non-TP golden model output.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "tp_only_forward.out"
    _run_ep_test_with_torchrun(2, "tp_only_forward", str(output))


# =============================================================================
# EP State Dict Update Tests
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_state_dict_update_2gpu(tmp_path_factory):
    """Test state dict correctness after optimizer step (ep=2, tp=2) on 2 GPUs.

    Verify:
    1. Forward + backward + optimizer.step()
    2. Gather EP weights to full state_dict
    3. Load to non-EP model, forward outputs should match.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "state_dict_update.out"
    _run_ep_test_with_torchrun(2, "state_dict_update", str(output))


# =============================================================================
# DTensor Checkpoint Roundtrip Tests
# =============================================================================


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_tp_dtensor_checkpoint_2gpu(tmp_path_factory):
    """Test DTensor checkpoint roundtrip for EP+TP (ep=2, tp=2) on 2 GPUs.

    Tests MoEWeightConverter methods:
    - split_expert_weights_dtensor(): 3D DTensor -> 2D DTensors
    - concatenate_expert_weights_dtensor(): 2D DTensors -> 3D DTensor

    Verify: to_hf() -> from_hf() roundtrip preserves DTensor weights.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_tp_dtensor_checkpoint.out"
    _run_ep_test_with_torchrun(2, "ep_tp_dtensor_checkpoint", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_ep_only_dtensor_checkpoint_2gpu(tmp_path_factory):
    """Test DTensor checkpoint roundtrip for EP only (ep=2, tp=1) on 2 GPUs.

    Tests MoEWeightConverter methods with EP-only configuration.

    Verify: to_hf() -> from_hf() roundtrip preserves DTensor weights.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "ep_only_dtensor_checkpoint.out"
    _run_ep_test_with_torchrun(2, "ep_only_dtensor_checkpoint", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_etp_dtensor_checkpoint_4gpu(tmp_path_factory):
    """Test DTensor checkpoint roundtrip for ETP (ep=2, tp=2, etp=2) on 4 GPUs.

    Tests MoEWeightConverter methods with ETP configuration
    (StridedShard + Shard placement).

    Verify: to_hf() -> from_hf() roundtrip preserves DTensor weights.
    """
    if current_platform.device_count() < 4:
        pytest.skip("This test requires 4 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "etp_dtensor_checkpoint.out"
    _run_ep_test_with_torchrun(4, "etp_dtensor_checkpoint", str(output))
