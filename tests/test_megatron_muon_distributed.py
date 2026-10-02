# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import torch


@pytest.mark.slow
@pytest.mark.multi_gpu
@pytest.mark.parametrize("tp", [1, 2])
def test_muon_three_updates_and_checkpoint_fourth_update_match(tmp_path, tp):
    """Exercise native Muon via the real engine and checkpoint manager on DP/TP."""
    model = os.environ.get("AREAL_MUON_TEST_MODEL")
    if not model:
        pytest.skip("Set AREAL_MUON_TEST_MODEL to a local Qwen2.5 model")
    if torch.cuda.device_count() < 2 * tp:
        pytest.skip(f"DP2/TP{tp} requires {2 * tp} GPUs")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={2 * tp}",
            "tests/torchrun/run_megatron_muon.py",
            "--model",
            model,
            "--backend",
            f"megatron:d2p1t{tp}",
            "--output",
            str(tmp_path),
        ],
        check=True,
        timeout=1800,
    )
