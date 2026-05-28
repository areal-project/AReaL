"""Integration tests for offload functionality in FSDP and Megatron engines using TMS."""

import multiprocessing
import os
import time
import traceback
from contextlib import contextmanager

import pytest
import torch.distributed as dist

from tests.utils import get_model_path

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import MegatronEngineConfig, OptimizerConfig, TrainEngineConfig
from areal.engine import FSDPEngine, MegatronEngine
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports
from areal.utils.offload import get_tms_env_vars

MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)


def _create_engine(engine_type: str):
    """Create FSDP/Megatron engine with TMS offload enabled."""
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(find_free_ports(1)[0]),
        }
    )

    config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name="test_offload",
        trial_name=f"{engine_type}_tms",
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        megatron=MegatronEngineConfig(),
    )

    alloc_mode = ModelAllocation.from_str("fsdp:d1p1t1")
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)

    if engine_type == "FSDP":
        engine = FSDPEngine(config)
    elif engine_type == "Megatron":
        engine = MegatronEngine(config)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")

    engine.create_process_group(alloc_mode.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)

    print(f"{engine_type} engine initialized")
    return engine


def _run_test(
    engine_type: str,
    min_memory_release_gb: float = 0.1,
    memory_tolerance: float = 0.1,
    warmup_rounds: int = 3,
    output_queue=None,
):
    """Function to run in subprocess. Creates engine and runs test."""
    try:
        print(f"[Subprocess] Starting test for {engine_type}...")

        engine = _create_engine(engine_type)

        try:
            _test_offload_and_onload(
                engine=engine,
                engine_name=engine_type,
                min_memory_release_gb=min_memory_release_gb,
                memory_tolerance=memory_tolerance,
                warmup_rounds=warmup_rounds,
            )
            if output_queue:
                output_queue.put(True)
        finally:
            engine.destroy()
            if dist.is_initialized():
                dist.destroy_process_group()

    except Exception as e:
        print(f"[Subprocess] Error: {e}")
        traceback.print_exc()
        if output_queue:
            output_queue.put(False)
        raise


# =============================================================================
# Multiprocessing Helpers
# =============================================================================


@contextmanager
def _tms_env_context():
    tms_env = get_tms_env_vars()
    os.environ.update(tms_env)
    yield


def _run_in_subprocess(target, kwargs):
    """Run function in a subprocess with TMS environment configured."""
    ctx = multiprocessing.get_context("spawn")
    output_queue = ctx.Queue()
    kwargs["output_queue"] = output_queue

    # Set env vars in parent process before spawning
    # Spawned process will inherit these variables
    with _tms_env_context():
        p = ctx.Process(target=target, kwargs=kwargs)
        p.start()
        p.join()

    # Check results
    if not output_queue.empty():
        success = output_queue.get()
        if not success:
            pytest.fail("Test failed in subprocess")
    else:
        if p.exitcode != 0:
            pytest.fail(f"Subprocess crashed with exit code {p.exitcode}")
        else:
            pytest.fail("Subprocess finished but returned no result")


def get_gpu_memory_allocated_gb() -> float:
    """Get currently allocated GPU memory in GB."""
    device = current_platform.current_device()
    allocated = current_platform.device_memory_used(device)
    return allocated / (1024**3)


def _test_offload_and_onload(
    engine,
    engine_name: str,
    min_memory_release_gb: float = 0.1,
    memory_tolerance: float = 0.1,
    warmup_rounds: int = 3,
):
    # Measure initial memory
    current_platform.synchronize()
    initial_memory_gb = get_gpu_memory_allocated_gb()
    print(f"[{engine_name}] Initial GPU memory: {initial_memory_gb:.2f} GB")

    # Warm up
    print(f"[{engine_name}] Running {warmup_rounds} warmup cycles...")
    for _ in range(warmup_rounds):
        engine.offload()
        engine.onload()
    current_platform.synchronize()

    # === Test Offload ===
    start_time = time.perf_counter()
    engine.offload()
    offload_time = time.perf_counter() - start_time

    current_platform.synchronize()
    memory_after_offload_gb = get_gpu_memory_allocated_gb()
    memory_released_gb = initial_memory_gb - memory_after_offload_gb

    print(
        f"[{engine_name}] After offload: {memory_after_offload_gb:.2f} GB "
        f"(released {memory_released_gb:.2f} GB in {offload_time:.3f}s)"
    )

    # Assert memory was released
    assert memory_released_gb > min_memory_release_gb, (
        f"Expected memory release > {min_memory_release_gb:.2f} GB, "
        f"but only {memory_released_gb:.2f} GB was released"
    )

    if offload_time > 0:
        offload_speed_gbps = memory_released_gb / offload_time
        print(f"[{engine_name}] Offload speed: {offload_speed_gbps:.2f} GB/s")

    # === Test Onload ===
    start_time = time.perf_counter()
    engine.onload()
    onload_time = time.perf_counter() - start_time

    current_platform.synchronize()
    memory_after_onload_gb = get_gpu_memory_allocated_gb()
    memory_restored_gb = memory_after_onload_gb - memory_after_offload_gb

    print(
        f"[{engine_name}] After onload: {memory_after_onload_gb:.2f} GB "
        f"(restored {memory_restored_gb:.2f} GB in {onload_time:.3f}s)"
    )

    # Memory should be restored to approximately initial level
    memory_diff = abs(memory_after_onload_gb - initial_memory_gb)
    tolerance = initial_memory_gb * memory_tolerance
    assert memory_diff < tolerance, (
        f"Memory not restored correctly: initial={initial_memory_gb:.2f} GB, "
        f"after_onload={memory_after_onload_gb:.2f} GB, diff={memory_diff:.2f} GB"
    )

    if onload_time > 0 and memory_restored_gb > 0:
        onload_speed_gbps = memory_restored_gb / onload_time
        print(f"[{engine_name}] Onload speed: {onload_speed_gbps:.2f} GB/s")


# =============================================================================
# Offload Tests
# =============================================================================


@pytest.mark.parametrize("engine_type", ["FSDP", "Megatron"])
@pytest.mark.slow
def test_engine_offload_and_onload(engine_type):
    """Test engine offload releases memory and onload recovers it correctly.

    This test validates:
    1. Memory is released during offload
    2. Transfer speed is reasonable
    3. Memory is restored correctly after onload

    Parametrized to test both FSDP and Megatron engines.
    """
    _run_in_subprocess(
        target=_run_test,
        kwargs={
            "engine_type": engine_type,
            "min_memory_release_gb": 0.1,
            "memory_tolerance": 0.1,
            "warmup_rounds": 3,
        },
    )
