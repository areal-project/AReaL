"""Context Parallelism (CP/Ulysses) tests for Archon Engine.

Run tests:
    pytest tests/experimental/archon/test_distributed_cp.py -v -m multi_gpu
"""

import subprocess

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_cp_test_with_torchrun(n_gpus: int, cp_size: int):
    """Run FSDP+CP (Ulysses) Archon forward test."""
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/experimental/archon/torchrun/run_cp_forward.py",
                f"--cp_size={cp_size}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_cp_forward_2gpu():
    """Test Archon Engine forward with CP (Ulysses) on 2 GPUs (dp=1, cp=2)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    _run_cp_test_with_torchrun(2, 2)
