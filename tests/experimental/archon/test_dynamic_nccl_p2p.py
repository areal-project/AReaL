"""Minimal NCCL P2P test for dynamic request-driven communication."""

import pytest
import torch

from tests.experimental.archon.utils import run_torchrun_test

from areal.infra.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_dynamic_nccl_p2p_mailbox():
    """Verify dynamic header-driven send/recv without a fixed global order."""
    if current_platform.device_count() < 3:
        pytest.skip("This test requires at least 3 GPUs")

    run_torchrun_test(
        "tests/experimental/archon/torchrun/run_dynamic_nccl_p2p.py",
        n_gpus=3,
    )
