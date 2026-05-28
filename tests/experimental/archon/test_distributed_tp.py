"""Tensor Parallelism (TP) tests for Archon Engine.

Run tests:
    pytest tests/experimental/archon/test_distributed_tp.py -v -m multi_gpu
"""

import subprocess

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_tp_test_with_torchrun(n_gpus: int, tp_size: int):
    """Run FSDP+TP Archon forward test."""
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/experimental/archon/torchrun/run_tp_forward.py",
                f"--tp_size={tp_size}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_tp_forward_2gpu():
    """Test Archon Engine forward with TP on 2 GPUs (dp=1, tp=2)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    _run_tp_test_with_torchrun(2, 2)
