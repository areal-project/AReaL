# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.slow
@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_megatron_mopd_cp2_scalar_reassembly():
    """Run the two-rank CP scalar-only reconstruction regression."""
    script = Path(__file__).with_name("run_megatron_mopd_cp2.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(script),
        ],
        check=True,
        timeout=120,
    )
