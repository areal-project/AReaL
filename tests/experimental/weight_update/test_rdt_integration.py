# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess

import pytest
import torch

from areal.infra.platforms import current_platform
from areal.infra.utils.proc import kill_process_tree
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="RDT tests require CUDA GPU (NIXL transport)"
)

# Project root so that torchrun workers can resolve `from tests.*` imports.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def _run_rdt_weight_transfer_test(n_gpus: int, test_type: str, output: str):
    """Run RDT distributed test via torchrun subprocess."""
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
            "tests/experimental/weight_update/torchrun/run_rdt_weight_transfer.py",
            f"--test_type={test_type}",
            f"--output={output}",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        env=env,
    )
    try:
        stdout, _ = proc.communicate(timeout=300)
        print(stdout)
    except BaseException:
        kill_process_tree(proc.pid)
        raise
    if proc.returncode != 0:
        pytest.fail(f"torchrun exited with code {proc.returncode}")

    with open(output) as f:
        result = f.read().strip()
    assert result == "Passed", f"Test failed: {result}"


# ---------------------------------------------------------------------------
# Distributed RDT tests via torchrun
# ---------------------------------------------------------------------------


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_rdt_weight_transfer_lifecycle_2gpu(tmp_path_factory):
    """Test full RDT weight transfer lifecycle."""
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "rdt_lifecycle.out"
    _run_rdt_weight_transfer_test(2, "rdt_weight_transfer_lifecycle", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("n_gpus", [4])
def test_rdt_weight_transfer_lifecycle_4gpu(n_gpus, tmp_path_factory):
    """Test RDT weight transfer with 4 GPUs."""
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    output = tmp_path_factory.mktemp("test_output") / f"rdt_lifecycle_{n_gpus}gpu.out"
    _run_rdt_weight_transfer_test(n_gpus, "rdt_weight_transfer_lifecycle", str(output))


# ---------------------------------------------------------------------------
# E2E RDT weight update: real FSDPEngine + SGLang server + gateway
# ---------------------------------------------------------------------------


def _get_test_model_path() -> str:
    local = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
    if os.path.isdir(local):
        return local
    return "Qwen/Qwen3-0.6B"


def _make_local_scheduler(tmp_path, name: str, gpu_devices: list[int]):
    from areal.infra.scheduler.local import LocalScheduler

    fileroot = tmp_path / f"{name}_fileroot"
    fileroot.mkdir(exist_ok=True)
    nr_root = tmp_path / f"{name}_name_resolve"
    nr_root.mkdir(exist_ok=True)

    return LocalScheduler(
        gpu_devices=gpu_devices,
        log_dir=str(tmp_path / f"{name}_logs"),
        experiment_name=f"test-rdt-{name}",
        trial_name="t0",
        fileroot=str(fileroot),
        nfs_record_root=str(nr_root),
    )


# Representative parameters spanning different fusion/sharding cases.
_VALIDATE_PARAM_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.27.self_attn.q_proj.weight",
    "model.norm.weight",
]


def _validate_weight_update_correctness(
    train_worker_urls: list[str],
    inf_worker_url: str,
    param_dir,
) -> None:
    """Fetch params from both sides via HTTP and compare bitwise."""
    import httpx

    n_train = len(train_worker_urls)
    print(
        f"\n[weight-validation] Fetching parameters from {n_train} training "
        f"worker(s) and 1 inference worker ..."
    )

    train_shard_paths = []
    for i, url in enumerate(train_worker_urls):
        p = str(param_dir / f"train_params_rank{i}.pt")
        train_shard_paths.append(p)
        resp = httpx.post(
            f"{url}/rdt/debug/get_parameters",
            json={"save_path": p, "names": _VALIDATE_PARAM_NAMES},
            timeout=120.0,
        )
        assert resp.status_code == 200, (
            f"get_parameters failed on training worker {i}: {resp.text}"
        )

    inf_path = str(param_dir / "infer_params.pt")
    resp = httpx.post(
        f"{inf_worker_url}/rdt/debug/get_parameters",
        json={"save_path": inf_path, "names": _VALIDATE_PARAM_NAMES},
        timeout=120.0,
    )
    assert resp.status_code == 200, (
        f"get_parameters failed on inference worker: {resp.text}"
    )

    infer_params = torch.load(inf_path, map_location="cpu", weights_only=True)
    train_shards = [
        torch.load(p, map_location="cpu", weights_only=True) for p in train_shard_paths
    ]

    print(f"[weight-validation] Comparing {len(_VALIDATE_PARAM_NAMES)} parameters ...")
    for name in _VALIDATE_PARAM_NAMES:
        assert name in infer_params, f"Inference missing param: {name}"
        for i, shard in enumerate(train_shards):
            assert name in shard, f"Training rank {i} missing param: {name}"

        # Reconstruct full training param from FSDP Shard(0) chunks
        if len(train_shards) > 1:
            full_train = torch.cat([s[name] for s in train_shards], dim=0)
        else:
            full_train = train_shards[0][name]

        torch.testing.assert_close(
            full_train,
            infer_params[name],
            rtol=0,
            atol=0,
            msg=f"Parameter mismatch after weight update: {name}",
        )
        print(
            f"[weight-validation]   {name}: OK "
            f"(shape={list(full_train.shape)}, dtype={full_train.dtype})"
        )

    print(
        f"[weight-validation] All {len(_VALIDATE_PARAM_NAMES)} parameters "
        f"match between training and inference"
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.ray
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_rdt_fsdp_e2e_weight_update(n_gpus, tmp_path_factory):
    """Full round trip: FSDPEngine → weight-update gateway → SGLang (RDT mode).

    Requires Ray cluster with NIXL tensor transport support.
    """
    import httpx

    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")

    # Require Ray cluster
    import ray

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.experimental.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.experimental.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    n_half = n_gpus // 2
    tmp = tmp_path_factory.mktemp("rdt_e2e")
    model_path = _get_test_model_path()

    scheduler = _make_local_scheduler(tmp, "rdt_e2e", gpu_devices=list(range(n_gpus)))

    # IW: SGLang with RDT backend
    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_half}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.experimental.inference_service.guard",
                env_vars={"AREAL_WEIGHT_UPDATE_BACKEND": "rdt"},
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    # TW: FSDP
    train_config = TrainEngineConfig(
        backend=f"fsdp:d{n_half}",
        experiment_name="test-rdt-e2e",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.experimental.training_service.guard",
                env_vars=dict(
                    NCCL_CUMEM_ENABLE="0",
                    NCCL_NVLS_ENABLE="0",
                    AREAL_WEIGHT_UPDATE_BACKEND="rdt",
                ),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.fsdp_engine.FSDPEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None

    try:
        # -- 1. SGLang IW (RDT backend) --
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        # Randomize IW weights
        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/rdt/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        # -- 2. FSDP TW --
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        train_ctrl.initialize(role="actor", ft_spec=ft_spec, wait=True)
        train_worker_urls = list(train_ctrl._worker_addrs)

        # -- 3. Weight update gateway --
        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(
                host="127.0.0.1",
                request_timeout=300.0,
            )
        )
        wu_ctrl.initialize()

        # -- 4. RDT weight update lifecycle --
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        wu_ctrl.connect(
            pair_name="test_rdt_e2e",
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            mode="rdt",
        )

        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1

        wu_ctrl.disconnect()

        # -- 5. Verify IW generation works --
        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=30.0,
        )
        assert gen_resp.status_code == 200, (
            f"Generation failed after weight update: {gen_resp.text}"
        )

        # -- 6. Validate TW ↔ IW parameter equality --
        _validate_weight_update_correctness(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
        )

    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)
