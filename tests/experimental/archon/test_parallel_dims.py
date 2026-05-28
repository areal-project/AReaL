"""Unit tests for ArchonParallelDims ETP (Expert Tensor Parallelism) configuration.

These tests verify the ETP configuration validation and mesh building logic
without requiring GPU or distributed environment.

Run tests:
    pytest tests/experimental/archon/test_parallel_dims.py -v
"""

import subprocess

import pytest
import torch

from areal.experimental.models.archon import ArchonParallelDims
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


class TestPPValidation:
    """Test PP (Pipeline Parallelism) constraint validation in ArchonParallelDims."""

    def test_dp_shard_auto_calculation_with_pp(self):
        """Test that dp_shard auto-calculation includes pp."""
        # world_size=8, pp=2, tp=2, cp=1 -> dp_shard = 8 / (2*2*1) = 2
        dims = ArchonParallelDims(pp=2, tp=2, cp=1, world_size=8)
        assert dims.dp_shard == 2

        # world_size=8, pp=1, tp=2, cp=1 -> dp_shard = 8 / (1*2*1) = 4
        dims = ArchonParallelDims(pp=1, tp=2, cp=1, world_size=8)
        assert dims.dp_shard == 4

        # world_size=8, pp=2, tp=2, cp=2 -> dp_shard = 8 / (2*2*2) = 1
        dims = ArchonParallelDims(pp=2, tp=2, cp=2, world_size=8)
        assert dims.dp_shard == 1

    def test_world_size_validation_with_pp(self):
        """Test that world_size validation includes pp."""
        # Valid: dp_shard * tp * cp * pp = 2 * 2 * 1 * 2 = 8
        dims = ArchonParallelDims(pp=2, dp_shard=2, tp=2, cp=1, world_size=8)
        assert dims.world_size == 8

        # Invalid: dp_shard * tp * cp * pp = 2 * 2 * 1 * 2 = 8 != 4
        with pytest.raises(
            ValueError, match="dp_shard .* tp .* cp .* pp must equal world_size"
        ):
            ArchonParallelDims(pp=2, dp_shard=2, tp=2, cp=1, world_size=4)

    def test_pp_enabled_flag(self):
        """Test pp_enabled property."""
        # pp=1 -> not enabled
        dims = ArchonParallelDims(pp=1, dp_shard=2, tp=2, cp=1, world_size=4)
        assert not dims.pp_enabled

        # pp=2 -> enabled
        dims = ArchonParallelDims(pp=2, dp_shard=2, tp=2, cp=1, world_size=8)
        assert dims.pp_enabled

    def test_context_and_model_parallel_size(self):
        """Test context_and_model_parallel_size property (cp * tp * pp)."""
        # cp=1, tp=2, pp=2 -> 1 * 2 * 2 = 4
        dims = ArchonParallelDims(pp=2, dp_shard=2, tp=2, cp=1, world_size=8)
        assert dims.context_and_model_parallel_size == 4

        # cp=2, tp=2, pp=1 -> 2 * 2 * 1 = 4
        dims = ArchonParallelDims(pp=1, dp_shard=2, tp=2, cp=2, world_size=8)
        assert dims.context_and_model_parallel_size == 4

        # cp=1, tp=1, pp=1 -> 1 * 1 * 1 = 1
        dims = ArchonParallelDims(pp=1, dp_shard=2, tp=1, cp=1, world_size=2)
        assert dims.context_and_model_parallel_size == 1


class TestPPWithEP:
    """Test PP combined with EP configurations."""

    def test_pp_with_ep_etp_disabled(self):
        """Test PP with EP when etp=1."""
        # pp=2, dp_shard=2, tp=2, cp=1, ep=4, etp=1, world_size=8
        # EP borrows from dp_shard * cp * tp = 2 * 1 * 2 = 4
        dims = ArchonParallelDims(
            pp=2, dp_shard=2, tp=2, cp=1, ep=4, etp=1, world_size=8
        )
        assert dims.pp_enabled
        assert dims.ep_enabled
        assert not dims.etp_enabled

    def test_pp_with_ep_etp_enabled(self):
        """Test PP with EP when etp=tp."""
        # pp=2, dp_shard=4, tp=2, cp=1, ep=4, etp=2, world_size=16
        # EP borrows from dp_shard * cp = 4 * 1 = 4
        dims = ArchonParallelDims(
            pp=2, dp_shard=4, tp=2, cp=1, ep=4, etp=2, world_size=16
        )
        assert dims.pp_enabled
        assert dims.ep_enabled
        assert dims.etp_enabled


class TestETPValidation:
    """Test ETP constraint validation in ArchonParallelDims."""

    def test_etp_must_be_1_or_tp(self):
        """Test that etp must be 1 or equal to tp."""
        # Valid: etp=1 (default)
        dims = ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=1, etp=1, world_size=4)
        assert dims.etp == 1
        assert not dims.etp_enabled

        # Valid: etp=tp
        dims = ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=2, etp=2, world_size=4)
        assert dims.etp == 2
        assert dims.etp_enabled

        # Invalid: etp not in {1, tp}
        with pytest.raises(ValueError, match="etp must be 1 or equal to tp"):
            ArchonParallelDims(dp_shard=2, tp=4, cp=1, ep=1, etp=2, world_size=8)

    def test_etp_1_ep_borrows_dp_cp_tp(self):
        """Test that when etp=1, EP borrows from dp_shard * cp * tp."""
        # Valid: ep=2, etp=1, ep divides cp*tp=2
        dims = ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=2, etp=1, world_size=4)
        assert dims.ep == 2
        assert dims.etp == 1
        assert not dims.etp_enabled

        # Invalid: ep=3 doesn't divide cp*tp=2
        with pytest.raises(ValueError, match="ep must be divisible by cp \\* tp"):
            ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=3, etp=1, world_size=4)

    def test_etp_tp_ep_borrows_dp_cp_only(self):
        """Test that when etp=tp, EP borrows from dp_shard * cp only (not tp)."""
        # Valid: ep=2, etp=2, ep divides dp_shard*cp=2
        dims = ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=2, etp=2, world_size=4)
        assert dims.ep == 2
        assert dims.etp == 2
        assert dims.etp_enabled

        # Invalid: ep=3 doesn't divide cp=2 (3 % 2 != 0)
        with pytest.raises(ValueError, match="ep must be divisible by cp"):
            ArchonParallelDims(dp_shard=2, tp=2, cp=2, ep=3, etp=2, world_size=8)

        # Invalid: dp_shard * cp not divisible by ep (2 * 1 = 2, 2 % 3 != 0)
        with pytest.raises(ValueError, match="dp_shard \\* cp must be divisible by ep"):
            ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=3, etp=2, world_size=4)


class TestETPMeshConfiguration:
    """Test ETP mesh dimension configurations."""

    def test_ep_only_no_etp(self):
        """Test EP only configuration (etp=1)."""
        # dp_shard=2, tp=1, cp=1, ep=2, etp=1 (2 GPUs)
        dims = ArchonParallelDims(dp_shard=2, tp=1, cp=1, ep=2, etp=1, world_size=2)
        assert dims.ep_enabled
        assert not dims.etp_enabled
        assert not dims.tp_enabled

    def test_ep_tp_no_etp(self):
        """Test EP+TP configuration with etp=1 (TP borrowed by EP)."""
        # dp_shard=1, tp=2, cp=1, ep=2, etp=1 (2 GPUs)
        # EP borrows from dp_shard * cp * tp = 1 * 1 * 2 = 2
        dims = ArchonParallelDims(dp_shard=1, tp=2, cp=1, ep=2, etp=1, world_size=2)
        assert dims.ep_enabled
        assert dims.tp_enabled
        assert not dims.etp_enabled  # etp=1, so ETP is not enabled

    def test_ep_tp_with_etp(self):
        """Test EP+TP configuration with etp=tp (ExpertTensorParallel)."""
        # dp_shard=2, tp=2, cp=1, ep=2, etp=2 (4 GPUs)
        # EP borrows from dp_shard * cp = 2 * 1 = 2
        dims = ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=2, etp=2, world_size=4)
        assert dims.ep_enabled
        assert dims.tp_enabled
        assert dims.etp_enabled

    def test_tp_only_no_ep(self):
        """Test TP only configuration (ep=1)."""
        # dp_shard=1, tp=2, cp=1, ep=1, etp=1 (2 GPUs)
        dims = ArchonParallelDims(dp_shard=1, tp=2, cp=1, ep=1, etp=1, world_size=2)
        assert not dims.ep_enabled
        assert dims.tp_enabled
        assert not dims.etp_enabled


class TestETPStrategySelection:
    """Test that correct strategy would be selected based on EP/TP/ETP config.

    Strategy Selection Table:
    | EP  | TP  | etp | Strategy              |
    |-----|-----|-----|-----------------------|
    | 1   | 1   | -   | None (Replicate)      |
    | 1   | >1  | -   | TensorParallel        |
    | >1  | 1   | -   | ExpertParallel        |
    | >1  | >1  | 1   | ExpertParallel        |
    | >1  | >1  | tp  | ExpertTensorParallel  |
    """

    def test_strategy_none_replicate(self):
        """EP=1, TP=1: No parallelism, weights replicated."""
        dims = ArchonParallelDims(dp_shard=2, tp=1, cp=1, ep=1, etp=1, world_size=2)
        assert not dims.ep_enabled
        assert not dims.tp_enabled
        assert not dims.etp_enabled

    def test_strategy_tensor_parallel(self):
        """EP=1, TP>1: TensorParallel for experts."""
        dims = ArchonParallelDims(dp_shard=1, tp=2, cp=1, ep=1, etp=1, world_size=2)
        assert not dims.ep_enabled
        assert dims.tp_enabled
        assert not dims.etp_enabled

    def test_strategy_expert_parallel_ep_only(self):
        """EP>1, TP=1: ExpertParallel only."""
        dims = ArchonParallelDims(dp_shard=2, tp=1, cp=1, ep=2, etp=1, world_size=2)
        assert dims.ep_enabled
        assert not dims.tp_enabled
        assert not dims.etp_enabled

    def test_strategy_expert_parallel_ep_tp_borrowed(self):
        """EP>1, TP>1, etp=1: ExpertParallel (TP borrowed by EP)."""
        dims = ArchonParallelDims(dp_shard=1, tp=2, cp=1, ep=2, etp=1, world_size=2)
        assert dims.ep_enabled
        assert dims.tp_enabled
        assert not dims.etp_enabled  # TP is borrowed, not used for experts

    def test_strategy_expert_tensor_parallel(self):
        """EP>1, TP>1, etp=tp: ExpertTensorParallel (2D sharding)."""
        dims = ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=2, etp=2, world_size=4)
        assert dims.ep_enabled
        assert dims.tp_enabled
        assert dims.etp_enabled  # Both EP and TP used for experts


class TestETPWorldSizeCalculation:
    """Test world size calculations with ETP configurations."""

    def test_world_size_ep_etp_4gpu(self):
        """Test 4 GPU configuration with EP+ETP."""
        # archon:(attn:d2t2|ffn:e2t2) -> 4 GPUs
        dims = ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=2, etp=2, world_size=4)
        assert dims.dp_shard * dims.tp * dims.cp == 4

    def test_world_size_ep_no_etp_4gpu(self):
        """Test 4 GPU configuration with EP only (etp=1)."""
        # archon:(attn:d2t2|ffn:e4t1) -> 4 GPUs
        # EP borrows from dp_shard * cp * tp = 2 * 1 * 2 = 4
        dims = ArchonParallelDims(dp_shard=2, tp=2, cp=1, ep=4, etp=1, world_size=4)
        assert dims.dp_shard * dims.tp * dims.cp == 4
        assert dims.ep == 4
        assert dims.etp == 1

    def test_world_size_ep_etp_8gpu(self):
        """Test 8 GPU configuration with EP+ETP."""
        # archon:(attn:d2t2|ffn:e4t2) -> 8 GPUs
        # EP borrows from dp_shard * cp = 4 * 1 = 4, ep=4
        dims = ArchonParallelDims(dp_shard=4, tp=2, cp=1, ep=4, etp=2, world_size=8)
        assert dims.dp_shard * dims.tp * dims.cp == 8
        assert dims.ep == 4
        assert dims.etp == 2


class TestETPMeshDimensions:
    """Test that ep_tp mesh is correctly 2D after flattening.

    This tests the fix for the ValueError:
    `placements` must have the same length as `device_mesh.ndim`!
    Found placements length: 2, and device_mesh.ndim: 3.

    The ep_tp mesh must be 2D [ep, tp] for ExpertTensorParallel to work.
    """

    @staticmethod
    def _run_parallel_dims_test(n_gpus: int, test_type: str, output: str):
        """Run parallel dims test with torchrun."""
        port = find_free_ports(1)[0]
        try:
            subprocess.run(
                [
                    "torchrun",
                    f"--nproc_per_node={n_gpus}",
                    "--nnodes=1",
                    "--master-addr=localhost",
                    f"--master_port={port}",
                    "tests/experimental/archon/torchrun/run_parallel_dims.py",
                    f"--test_type={test_type}",
                    f"--output={output}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Test failed with error: {e.stderr}")

        with open(output) as f:
            result = f.read().strip()
        assert result == "Passed", f"Test failed: {result}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.multi_gpu
    @pytest.mark.slow
    def test_ep_mesh_when_etp_disabled_2gpu(self, tmp_path_factory):
        """Test that ep mesh is 1D and ep_tp mesh is None when etp=1 (2 GPU)."""
        if current_platform.device_count() < 2:
            pytest.skip("This test requires 2 GPUs")
        output = tmp_path_factory.mktemp("test_output") / "ep_mesh.out"
        self._run_parallel_dims_test(2, "ep_mesh", str(output))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.multi_gpu
    @pytest.mark.slow
    def test_etp_mesh_is_2d_4gpu(self, tmp_path_factory):
        """Test that ep_tp mesh is 2D with dimensions [ep, tp] when etp=tp (4 GPU)."""
        if current_platform.device_count() < 4:
            pytest.skip("This test requires 4 GPUs")
        output = tmp_path_factory.mktemp("test_output") / "etp_mesh.out"
        self._run_parallel_dims_test(4, "etp_mesh", str(output))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.multi_gpu
    @pytest.mark.slow
    def test_pp_mesh_4gpu(self, tmp_path_factory):
        """Test that PP mesh dimension is correctly included (4 GPU)."""
        if current_platform.device_count() < 4:
            pytest.skip("This test requires 4 GPUs")
        output = tmp_path_factory.mktemp("test_output") / "pp_mesh.out"
        self._run_parallel_dims_test(4, "pp_mesh", str(output))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
