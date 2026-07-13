import subprocess

import pytest

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


def _run_test_with_torchrun(alloc_mode: str, output: str):
    port = find_free_ports(1)[0]
    n_gpus = ModelAllocation.from_str(alloc_mode).parallel.world_size
    try:
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_fsdp_memory_efficient_lora.py",
                f"--backend={alloc_mode}",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}, {e.stdout}")
    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


@pytest.mark.slow
def test_fsdp_memory_efficient_lora(tmp_path_factory):
    if current_platform.device_count() < 1:
        pytest.skip("Test requires at least 1 GPU")
    output = tmp_path_factory.mktemp("test_output") / "fsdp_memory_efficient_lora.out"
    _run_test_with_torchrun("fsdp:d1t1c1", str(output))
