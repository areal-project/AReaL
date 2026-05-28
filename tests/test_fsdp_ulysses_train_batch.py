import subprocess

import pytest

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(n_gpus: int):
    port = find_free_ports(1)[0]
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_fsdp_ulysses_train_batch.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}")


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("world_size", [2])
def test_fsdp_ulysses_train_batch_2gpu(world_size):
    if current_platform.device_count() < world_size:
        pytest.skip(f"This test requires {world_size} gpus")
    _run_test_with_torchrun(world_size)


@pytest.mark.slow
@pytest.mark.parametrize("world_size", [1])
def test_fsdp_ulysses_train_batch_1gpu(world_size):
    _run_test_with_torchrun(world_size)
