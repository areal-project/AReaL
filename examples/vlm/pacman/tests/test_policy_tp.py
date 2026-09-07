# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
def test_token_set_tp_statistics_and_gradients_match_unsharded():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("run_policy_tp.py")),
        ],
        check=True,
        timeout=180,
    )
