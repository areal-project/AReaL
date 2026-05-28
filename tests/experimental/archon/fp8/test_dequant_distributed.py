import subprocess

import pytest
import torch

from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _run_torchrun_test(
    script: str,
    n_gpus: int,
    output_file: str,
):
    port = find_free_ports(1)[0]
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        script,
        f"--output={output_file}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(f"--- STDOUT ---\n{result.stdout}")
    if result.stderr:
        print(f"--- STDERR ---\n{result.stderr}")

    if result.returncode != 0:
        pytest.fail(
            f"torchrun exited with code {result.returncode}\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    with open(output_file) as f:
        content = f.read().strip()
    if content != "Passed":
        pytest.fail(
            f"Test wrote '{content}' to output (expected 'Passed').\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )


class TestShardedDequant:
    def test_sharded_dequant_2gpu(self, tmp_path):
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            pytest.skip("Need at least 2 GPUs")
        _run_torchrun_test(
            "tests/experimental/archon/fp8/torchrun/run_sharded_dequant.py",
            min(n_gpus, 2),
            str(tmp_path / "result.out"),
        )

    def test_sharded_dequant_4gpu(self, tmp_path):
        n_gpus = torch.cuda.device_count()
        if n_gpus < 4:
            pytest.skip("Need at least 4 GPUs")
        _run_torchrun_test(
            "tests/experimental/archon/fp8/torchrun/run_sharded_dequant.py",
            4,
            str(tmp_path / "result.out"),
        )
