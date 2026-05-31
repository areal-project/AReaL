# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.infra.utils.proc import kill_process_tree
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

# Project root so that torchrun workers can resolve `from tests.*` imports.
# pytest adds "." via pyproject.toml `pythonpath`, but subprocesses don't inherit that.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _run_weight_update_test(n_gpus: int, test_type: str, output: str):
    port = find_free_ports(1)[0]
    env = os.environ.copy()
    env["PYTHONPATH"] = _PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        [
            "torchrun",
            f"--nproc_per_node={n_gpus}",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master_port={port}",
            "tests/experimental/weight_update/torchrun/run_fp8_weight_transfer.py",
            f"--test_type={test_type}",
            f"--output={output}",
        ],
        text=True,
        stderr=sys.stdout,
        stdout=sys.stdout,
        env=env,
    )
    try:
        proc.wait()
    except BaseException:
        kill_process_tree(proc.pid)
        raise
    if proc.returncode != 0:
        pytest.fail(f"torchrun exited with code {proc.returncode}")

    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_fp8_weight_transfer_2gpu(tmp_path_factory):
    """Test FP8 block-wise quantized weight transfer over NCCL with 2 GPUs.

    Rank 0 quantizes a BF16 weight to FP8, broadcasts the FP8 tensor and
    per-block scale to rank 1, plus a non-quantized 1D norm weight.
    Rank 1 verifies all tensors match exactly.
    """
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "fp8_weight_transfer.out"
    _run_weight_update_test(2, "fp8_weight_transfer", str(output))
