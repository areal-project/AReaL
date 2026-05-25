"""Distributed tests for sglang PP pipeline parallelism.

These tests verify that per-PP-rank NCCL weight update groups work correctly
in a multi-GPU distributed setting for all three training engine types:
Megatron, FSDP, and Archon.

Test matrix:
  E2E training tests (4 GPUs each, follow test_examples.py pattern):
  - test_pp_e2e_train[megatron]:  megatron:d1p2t2 + sglang:d1p2t2
  - test_pp_e2e_train[fsdp]:      fsdp:d2p1t2    + sglang:d1p2t2
  - test_pp_e2e_train[archon]:   archon:d2p1t2   + sglang:d1p2t2

  Torchrun unit tests (4 GPUs each):
  - test_pp_weight_group_init:   PP=2 TP=2 group init (per engine)
  - test_pp_weight_sync:         PP=2 TP=2 weight sync (per engine)

Requires 4 GPUs to run.
"""

import os
import re
import signal
import subprocess
import sys
import threading

import pytest

from tests.utils import get_dataset_path, get_model_path

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging
from areal.utils.network import find_free_ports

logger = logging.getLogger("TestSglangPPDistributed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENGINE_TYPES = ["megatron", "fsdp", "archon"]

SUCCESS_PATTERN = re.compile(r"Epoch 1/\d+ Step 1/\d+ Train step 1/\d+ done\.")

# Per-engine configuration for e2e tests.
# Each entry maps to one of the 3 verified working commands:
#   megatron: gsm8k_grpo_megatron.yaml, actor=megatron:d1p2t2, rollout=sglang:d1p2t2
#   fsdp:     gsm8k_grpo.yaml,          actor=fsdp:d2p1t2,     rollout=sglang:d1p2t2
#   archon:   gsm8k_grpo.yaml,          actor=archon:d2p1t2,   rollout=sglang:d1p2t2
ENGINE_CONFIGS = {
    "megatron": {
        "config": "examples/math/gsm8k_grpo_megatron.yaml",
        "actor_backend": "megatron:d1p2t2",
        "rollout_backend": "sglang:d1p2t2",
    },
    "fsdp": {
        "config": "examples/math/gsm8k_grpo.yaml",
        "actor_backend": "fsdp:d2p1t2",
        "rollout_backend": "sglang:d1p2t2",
    },
    "archon": {
        "config": "examples/math/gsm8k_grpo.yaml",
        "actor_backend": "archon:d2p1t2",
        "rollout_backend": "sglang:d1p2t2",
    },
}


# ===================================================================== #
#  E2E training tests (following test_examples.py pattern)              #
# ===================================================================== #


async def run_example(
    example_file: str,
    config_name: str,
    *additional_args,
    timeout: int = 480,
    success_pattern=SUCCESS_PATTERN,
) -> bool:
    """Run a single example in single-controller mode and return the result.

    This is the same pattern used in test_examples.py.
    """
    import asyncio
    import time

    cmd = [
        "python3",
        example_file,
        "--config",
        config_name,
    ]
    cmd += list(additional_args)

    logger.info(f"Running: {' '.join(cmd)}")

    success = False
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    start_time = time.monotonic()

    while True:
        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=0.1)
                if not line:
                    break
                line = line.decode().rstrip()
                if line:
                    logger.info(f"[Example Output] {line}")
                success = bool(success_pattern.search(line))
                if success:
                    break
            except (TimeoutError, ValueError):
                break

        if success:
            logger.info(f"✓ {example_file} with config {config_name} - SUCCESS")
            process.send_signal(signal.SIGINT)
            break

        try:
            return_code = await asyncio.wait_for(process.wait(), timeout=0.01)
            logger.error(f"Process terminated unexpectedly. Return code: {return_code}")
            break
        except TimeoutError:
            pass

        if (time.monotonic() - start_time) > timeout:
            logger.error("Process timed out without successful result, terminating...")
            process.send_signal(signal.SIGINT)
            break

    kill_process_tree(process.pid)
    return success


@pytest.mark.sglang
@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_pp_e2e_train(tmp_path_factory, engine_type):
    """End-to-end PP training test for each engine type.

    Runs one of the 3 verified working commands:
      megatron: gsm8k_grpo_megatron.yaml + megatron:d1p2t2 + sglang:d1p2t2
      fsdp:     gsm8k_grpo.yaml          + fsdp:d2p1t2     + sglang:d1p2t2
      archon:   gsm8k_grpo.yaml          + archon:d2p1t2   + sglang:d1p2t2

    All use 4 GPUs.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")

    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    cfg = ENGINE_CONFIGS[engine_type]
    example_file = "examples/math/gsm8k_rl.py"
    config_name = cfg["config"]

    success = run_async_task(
        run_example,
        example_file,
        config_name,
        f"rollout.backend={cfg['rollout_backend']}",
        f"actor.backend={cfg['actor_backend']}",
        "gconfig.n_samples=4",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=2",
        "valid_dataset.batch_size=2",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=4",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        "+total_train_steps=2",
        timeout=900,
    )
    assert success, f"PP e2e training failed for engine_type={engine_type}"


# ===================================================================== #
#  Torchrun helper utilities (for unit-level distributed tests)         #
# ===================================================================== #


def _get_alloc_mode(engine_type: str, dp: int, pp: int, tp: int) -> str:
    """Build a valid allocation mode string for the given engine type.

    FSDP training does not support pipeline parallelism (PP > 1).
    When PP > 1 is requested for FSDP, the PP dimension is folded into DP
    so that the total GPU count remains the same (dp * pp * tp).
    """
    if engine_type == "fsdp" and pp > 1:
        return f"fsdp:d{dp * pp}t{tp}"
    return f"{engine_type}:d{dp}p{pp}t{tp}"


def _reader_thread(pipe, collected, prefix):
    """Read lines from *pipe*, print with a prefix, and append to *collected*."""
    try:
        for line in pipe:
            sys.stdout.write(f"  [{prefix}] {line}")
            sys.stdout.flush()
            collected.append(line)
    except ValueError:
        pass


def _kill_process_tree_torchrun(proc):
    """Send SIGKILL to the process group rooted at *proc*."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()


def _run_test_with_torchrun(
    alloc_mode: str,
    test_type: str,
    output: str,
    gen_pp_size: int = 1,
    engine_type: str = "megatron",
):
    """Launch the distributed test worker under torchrun."""
    port = find_free_ports(1)[0]
    n_gpus = ModelAllocation.from_str(alloc_mode).parallel.world_size

    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        "tests/torchrun/run_sglang_pp_weight_sync.py",
        f"--backend={alloc_mode}",
        f"--output={output}",
        f"--test_type={test_type}",
        f"--gen_pp_size={gen_pp_size}",
        f"--engine_type={engine_type}",
    ]

    tag = f"{engine_type}/{test_type}"
    print(f"\n{'=' * 60}", flush=True)
    print(f"[torchrun] {tag}  alloc={alloc_mode}  gpus={n_gpus}", flush=True)
    print(f"[torchrun] cmd: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 60}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    collected: list[str] = []
    reader = threading.Thread(
        target=_reader_thread, args=(proc.stdout, collected, tag), daemon=True
    )
    reader.start()

    try:
        returncode = proc.wait(timeout=300)
        reader.join(timeout=5)

        if returncode != 0:
            tail = "".join(collected[-50:])
            pytest.fail(f"torchrun failed (returncode={returncode}):\n{tail}")
    except subprocess.TimeoutExpired:
        _kill_process_tree_torchrun(proc)
        reader.join(timeout=5)
        tail = "".join(collected[-50:])
        pytest.fail(f"torchrun timed out after 300s. Last output:\n{tail}")
    except Exception:
        _kill_process_tree_torchrun(proc)
        raise

    with open(output) as f:
        content = f.read().strip()
    assert content == "Passed", f"Test worker reported: {content}"


# ===================================================================== #
#  PP=2, TP=2 group initialization (4 GPUs)                             #
# ===================================================================== #


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_pp_weight_group_init(tmp_path_factory, engine_type):
    """Test per-PP-rank NCCL group initialization with PP=2, TP=2.

    Verifies:
      - Each PP rank gets a distinct group name ("update_weight_group_{pp_rank}").
      - is_pipeline_parallel_head detection is correct for each rank.
      - Per-PP-rank world sizes are computed correctly.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")

    alloc_mode = _get_alloc_mode(engine_type, dp=1, pp=2, tp=2)
    output = (
        tmp_path_factory.mktemp("test_output") / f"pp2_tp2_group_init_{engine_type}.out"
    )
    _run_test_with_torchrun(
        alloc_mode=alloc_mode,
        test_type="group_init",
        output=str(output),
        gen_pp_size=2,
        engine_type=engine_type,
    )


# ===================================================================== #
#  PP=2, TP=2 weight sync (4 GPUs)                                      #
# ===================================================================== #


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("engine_type", ENGINE_TYPES)
def test_pp_weight_sync(tmp_path_factory, engine_type):
    """Test weight sync with PP=2, TP=2 (4 GPUs).

    Verifies:
      - PP ranks hold different parameter name sets.
      - Weight sync completes without errors.
    """
    if current_platform.device_count() < 4:
        pytest.skip("Requires at least 4 GPUs")

    alloc_mode = _get_alloc_mode(engine_type, dp=1, pp=2, tp=2)
    output = tmp_path_factory.mktemp("test_output") / f"pp2_tp2_sync_{engine_type}.out"
    _run_test_with_torchrun(
        alloc_mode=alloc_mode,
        test_type="weight_sync",
        output=str(output),
        gen_pp_size=2,
        engine_type=engine_type,
    )
