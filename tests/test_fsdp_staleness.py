# SPDX-License-Identifier: Apache-2.0

"""Real optimizer tests using an offline tiny Qwen3 checkpoint by default."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.slow
@pytest.mark.multi_gpu
def test_fsdp_staleness_real_optimizer_preserves_empty_updates(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs and NCCL")
    model = os.environ.get("AREAL_TEST_MODEL_PATH")
    if model is None:
        from tests.torchrun.run_ppo_staleness import create_tiny_checkpoint

        model = str(tmp_path / "model")
        create_tiny_checkpoint(Path(model))
    elif not Path(model).is_dir():
        pytest.fail("AREAL_TEST_MODEL_PATH must be a local HF checkpoint")
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "tests/torchrun/run_ppo_staleness.py",
            "--model",
            model,
            "--output",
            str(tmp_path),
        ],
        check=True,
        timeout=600,
        cwd=repo,
        env=env,
    )
