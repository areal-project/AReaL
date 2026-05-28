"""Data Parallelism (DP/FSDP) tests for Archon Engine.

Run tests:
    pytest tests/experimental/archon/test_distributed_dp.py -v -m multi_gpu
"""

import pytest
import torch

from tests.experimental.archon.utils import run_torchrun_test

from areal.infra.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_archon_forward_dp_2gpu():
    """Test Archon Engine forward_batch with Data Parallelism (2 GPUs)."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    run_torchrun_test(
        "tests/experimental/archon/torchrun/run_forward.py",
        n_gpus=2,
    )
