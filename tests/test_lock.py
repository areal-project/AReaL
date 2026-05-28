import subprocess

import pytest

from areal.utils.network import find_free_ports


def _run_lock_test(world_size: int, backend: str = "nccl", iters: int = 10):
    port = find_free_ports(1)[0]
    cmd = [
        "torchrun",
        f"--nproc_per_node={world_size}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        "tests/torchrun/run_lock.py",
        "--backend",
        backend,
        "--iters",
        str(iters),
        "--timeout",
        "5.0",
        "--hold-time",
        "0.001",
        "--backoff",
        "0.005",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            "\n".join(
                [
                    "Distributed lock test failed",
                    f"Command: {' '.join(cmd)}",
                    f"Return code: {result.returncode}",
                    f"STDOUT:\n{result.stdout}",
                    f"STDERR:\n{result.stderr}",
                ]
            )
        )


@pytest.mark.parametrize("world_size", [1])
def test_distributed_lock_single_rank(world_size):
    _run_lock_test(world_size)


@pytest.mark.multi_gpu
@pytest.mark.parametrize("world_size", [2])
def test_distributed_lock_multi_rank(world_size):
    _run_lock_test(world_size)
