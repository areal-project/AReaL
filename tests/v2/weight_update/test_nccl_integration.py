# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import httpx
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
            "tests/v2/weight_update/torchrun/run_nccl_weight_transfer.py",
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
def test_nccl_group_init_2gpu(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "nccl_group_init.out"
    _run_weight_update_test(2, "nccl_group_init", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_batch_isend_irecv_2gpu(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "batch_isend_irecv.out"
    _run_weight_update_test(2, "batch_isend_irecv", str(output))


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_weight_transfer_lifecycle_2gpu(tmp_path_factory):
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")
    output = tmp_path_factory.mktemp("test_output") / "weight_transfer_4gpu.out"
    _run_weight_update_test(2, "weight_transfer_lifecycle", str(output))


# ---------------------------------------------------------------------------
# E2E awex weight update: real FSDPEngine + SGLang server + gateway
# ---------------------------------------------------------------------------


def _get_test_model_path() -> str:
    local = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
    if os.path.isdir(local):
        return local
    return "Qwen/Qwen3-0.6B"


def _get_test_moe_model_path() -> str:
    local = "/storage/openpsi/models/Qwen__Qwen3-30B-A3B/"
    if os.path.isdir(local):
        return local
    return "Qwen/Qwen3-30B-A3B"


def _make_truncated_moe_model(tmp_path, num_layers: int = 4) -> str:
    import glob
    import json
    import shutil

    src = _get_test_moe_model_path()
    dst = str(tmp_path / "truncated_moe")
    os.makedirs(dst, exist_ok=True)

    with open(os.path.join(src, "config.json")) as f:
        config = json.load(f)
    config["num_hidden_layers"] = num_layers
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(config, f)

    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
    ):
        src_file = os.path.join(src, fname)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst)

    for weight_file in glob.glob(os.path.join(src, "*.safetensors")) + glob.glob(
        os.path.join(src, "*.bin")
    ):
        os.symlink(weight_file, os.path.join(dst, os.path.basename(weight_file)))

    for index_file in glob.glob(
        os.path.join(src, "*.safetensors.index.json")
    ) + glob.glob(os.path.join(src, "*.bin.index.json")):
        shutil.copy2(index_file, dst)

    return dst


def _make_local_scheduler(tmp_path, name: str, gpu_devices: list[int]):
    from areal.infra.scheduler.local import LocalScheduler

    fileroot = tmp_path / f"{name}_fileroot"
    fileroot.mkdir(exist_ok=True)
    nr_root = tmp_path / f"{name}_name_resolve"
    nr_root.mkdir(exist_ok=True)

    return LocalScheduler(
        gpu_devices=gpu_devices,
        log_dir=str(tmp_path / f"{name}_logs"),
        experiment_name=f"test-awex-{name}",
        trial_name="t0",
        fileroot=str(fileroot),
        nfs_record_root=str(nr_root),
    )


# Representative parameters spanning different fusion/sharding cases.
# Covers QKV unfusing (q/k/v_proj), gate_up unfusing (gate/up_proj),
# a deeper layer, and a small non-attention weight (layer norm).
_VALIDATE_PARAM_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.27.self_attn.q_proj.weight",
    "model.norm.weight",
]

# Qwen3-30B-A3B (truncated to 4 layers) MoE validation params.
# This model has no shared experts — pure MoE with 128 routed experts.
_VALIDATE_PARAM_NAMES_MOE = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.1.mlp.experts.0.gate_proj.weight",
    "model.layers.1.mlp.experts.0.up_proj.weight",
    "model.layers.1.mlp.experts.0.down_proj.weight",
    "model.layers.3.self_attn.q_proj.weight",
    "model.norm.weight",
]


def _validate_weight_update_correctness(
    train_worker_urls: list[str],
    inf_worker_url: str,
    param_dir,
) -> None:
    """Fetch params from both sides via HTTP and compare bitwise.

    Training workers return FSDP local shards (Shard(0)), so we concatenate
    along dim 0 to reconstruct the full parameter.  Inference (TP=1) returns
    full parameters directly.
    """
    n_train = len(train_worker_urls)
    print(
        f"\n[weight-validation] Fetching parameters from {n_train} training "
        f"worker(s) and 1 inference worker …"
    )

    train_shard_paths = []
    for i, url in enumerate(train_worker_urls):
        p = str(param_dir / f"train_params_rank{i}.pt")
        train_shard_paths.append(p)
        resp = httpx.post(
            f"{url}/awex/debug/get_parameters",
            json={"save_path": p, "names": _VALIDATE_PARAM_NAMES},
            timeout=120.0,
        )
        assert resp.status_code == 200, (
            f"get_parameters failed on training worker {i}: {resp.text}"
        )

    inf_path = str(param_dir / "infer_params.pt")
    resp = httpx.post(
        f"{inf_worker_url}/awex/debug/get_parameters",
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

    print(f"[weight-validation] Comparing {len(_VALIDATE_PARAM_NAMES)} parameters …")
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
        f"match between training and inference ✓"
    )


def _validate_weight_update_correctness_megatron(
    train_worker_urls: list[str],
    inf_worker_url: str,
    param_dir,
    tag: str = "megatron",
    param_names: list[str] | None = None,
) -> None:
    import concurrent.futures

    names = param_names or _VALIDATE_PARAM_NAMES
    n_train = len(train_worker_urls)
    print(
        f"\n[weight-validation] Fetching parameters from {n_train} Megatron "
        f"worker(s) and 1 inference worker …"
    )

    train_paths = [
        str(param_dir / f"{tag}_train_params_rank{i}.pt") for i in range(n_train)
    ]

    def _fetch_train(args):
        i, url, p = args
        resp = httpx.post(
            f"{url}/awex/debug/get_parameters",
            json={"save_path": p, "names": names},
            timeout=120.0,
        )
        assert resp.status_code == 200, (
            f"get_parameters failed on training worker {i}: {resp.text}"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_train) as pool:
        list(
            pool.map(
                _fetch_train,
                [
                    (i, url, p)
                    for i, (url, p) in enumerate(zip(train_worker_urls, train_paths))
                ],
            )
        )

    inf_path = str(param_dir / f"{tag}_infer_params.pt")
    resp = httpx.post(
        f"{inf_worker_url}/awex/debug/get_parameters",
        json={"save_path": inf_path, "names": names},
        timeout=120.0,
    )
    assert resp.status_code == 200, (
        f"get_parameters failed on inference worker: {resp.text}"
    )

    infer_params = torch.load(inf_path, map_location="cpu", weights_only=True)

    # Union params across all training ranks: with PP each rank owns a disjoint
    # subset of layers, so we need all ranks to cover the validate param names.
    train_params: dict[str, torch.Tensor] = {}
    for p in train_paths:
        train_params.update(torch.load(p, map_location="cpu", weights_only=True))

    print(f"[weight-validation] Comparing {len(names)} parameters …")
    for name in names:
        assert name in infer_params, f"Inference missing param: {name}"
        assert name in train_params, (
            f"No training rank owns param: {name}. "
            f"Available: {sorted(train_params.keys())[:5]}…"
        )

        torch.testing.assert_close(
            train_params[name],
            infer_params[name],
            rtol=0,
            atol=0,
            msg=f"Parameter mismatch after weight update: {name}",
        )
        print(
            f"[weight-validation]   {name}: OK "
            f"(shape={list(train_params[name].shape)}, dtype={train_params[name].dtype})"
        )

    print(
        f"[weight-validation] All {len(names)} parameters "
        f"match between Megatron training and inference ✓"
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_awex_fsdp_e2e_weight_update(n_gpus, tmp_path_factory):
    """Full round trip: FSDPEngine (pure DP) → weight-update gateway → SGLang.

    A single :class:`LocalScheduler` owns all *n_gpus* devices.  Inference
    is initialized first so the round-robin allocator assigns the first half
    of the GPUs to SGLang and the second half to FSDP training — mirroring
    :class:`PPOTrainer` where one scheduler serves every engine.

    Orchestrates connect → update_weights → disconnect, passing the
    training *worker* URLs (not the training gateway) to the weight-update
    service since the awex blueprint lives on the workers.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    n_half = n_gpus // 2
    tmp = tmp_path_factory.mktemp("awex_e2e")
    model_path = _get_test_model_path()

    scheduler = _make_local_scheduler(tmp, "e2e", gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_half}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=f"fsdp:d{n_half}",
        experiment_name="test-awex-e2e",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
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
        # -- 1. SGLang via inference controller ----------------------------
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        # Randomize inference weights so the transfer is NOT a no-op.
        # Without this, both sides load from the same checkpoint and the
        # comparison would trivially pass even if the transfer never happened.
        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        # -- 2. FSDP engine via training controller -------------------------
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        train_ctrl.initialize(role="actor", ft_spec=ft_spec, wait=True)
        train_worker_urls = list(train_ctrl._worker_addrs)

        # -- 3. Weight update gateway -------------------------------------
        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(
                host="127.0.0.1",
                request_timeout=300.0,
            )
        )
        wu_ctrl.initialize()

        # -- 4. Weight update lifecycle -----------------------------------
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        wu_ctrl.connect(
            pair_name="test_e2e",
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
        )

        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1

        wu_ctrl.disconnect()

        # -- 5. Verify inference server still works post-update -----------
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

        # -- 6. Validate training ↔ inference parameter equality ----------
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


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_awex_megatron_dp_e2e_weight_update(n_gpus, tmp_path_factory):
    """Full round trip: MegatronEngine (pure DP) → weight-update gateway → SGLang.

    Each training rank holds a full copy of every parameter. Validation unions
    params across all training ranks and compares bitwise against SGLang.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    n_half = n_gpus // 2
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{n_half}",
        pair_name="test_megatron_dp_e2e",
        tag="megatron_dp_e2e",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,tp_size",
    [(4, 2), (8, 2), (8, 4)],
    ids=["4gpu-dp1tp2", "8gpu-dp2tp2", "8gpu-dp1tp4"],
)
def test_awex_megatron_dp_tp_e2e_weight_update(n_gpus, tp_size, tmp_path_factory):
    """Full round trip: MegatronEngine (DP+TP) → weight-update gateway → SGLang.

    TP ranks within a DP group each hold the same full parameter after
    all_gather_param. dp_replicated=True tells awex only one rank per group
    needs to send, avoiding redundant transfers.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    n_infer = n_gpus // 2
    n_train = n_gpus - n_infer
    dp_size = n_train // tp_size
    if dp_size < 1:
        pytest.skip(f"Not enough GPUs for dp={dp_size} tp={tp_size}")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{dp_size}t{tp_size}",
        pair_name=f"test_megatron_dp{dp_size}tp{tp_size}",
        tag=f"megatron_dp{dp_size}tp{tp_size}",
        tmp_path_factory=tmp_path_factory,
    )


def _run_megatron_awex_e2e(
    *,
    n_gpus: int,
    backend: str,
    pair_name: str,
    tag: str,
    tmp_path_factory,
    model_path: str | None = None,
    validate_param_names: list[str] | None = None,
    init_from_scratch: bool = False,
):
    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    n_infer = n_gpus // 2
    tmp = tmp_path_factory.mktemp(tag)
    model_path = model_path or _get_test_model_path()
    scheduler = _make_local_scheduler(tmp, tag, gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_infer}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=backend,
        experiment_name=f"test-awex-{tag}",
        trial_name="t0",
        path=model_path,
        init_from_scratch=init_from_scratch,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.megatron_engine.MegatronLMEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None
    try:
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        train_ctrl.initialize(
            role="actor",
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=2
            ),
            wait=True,
        )
        train_worker_urls = list(train_ctrl._worker_addrs)

        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(host="127.0.0.1", request_timeout=300.0)
        )
        wu_ctrl.initialize()
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        wu_ctrl.connect(
            pair_name=pair_name,
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
        )
        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1
        wu_ctrl.disconnect()

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

        _validate_weight_update_correctness_megatron(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
            tag=tag,
            param_names=validate_param_names,
        )
    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,pp_size",
    [(4, 2), (8, 4)],
    ids=["4gpu-dp1pp2", "8gpu-dp1pp4"],
)
def test_awex_megatron_pp_e2e_weight_update(n_gpus, pp_size, tmp_path_factory):
    """Full round trip: MegatronEngine (pure PP) → weight-update gateway → SGLang.

    Each PP stage owns a disjoint subset of layers. Validation unions params
    across all PP ranks to reconstruct the full parameter set for comparison.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d1p{pp_size}",
        pair_name=f"test_megatron_pp{pp_size}",
        tag=f"megatron_pp{pp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,dp_size,pp_size",
    [(8, 2, 2)],
    ids=["8gpu-dp2pp2"],
)
def test_awex_megatron_dp_pp_e2e_weight_update(
    n_gpus, dp_size, pp_size, tmp_path_factory
):
    """Full round trip: MegatronEngine (DP+PP) → weight-update gateway → SGLang.

    Combines data parallelism (multiple replicas) with pipeline parallelism
    (each replica split across PP stages). Each PP stage of each DP replica
    sends its own layer subset independently.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{dp_size}p{pp_size}",
        pair_name=f"test_megatron_dp{dp_size}pp{pp_size}",
        tag=f"megatron_dp{dp_size}pp{pp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,pp_size,tp_size",
    [(8, 2, 2)],
    ids=["8gpu-dp1pp2tp2"],
)
def test_awex_megatron_pp_tp_e2e_weight_update(
    n_gpus, pp_size, tp_size, tmp_path_factory
):
    """Full round trip: MegatronEngine (PP+TP) → weight-update gateway → SGLang.

    PP splits layers across stages; TP shards each stage's weights across ranks.
    Both dp_replicated=True (TP) and disjoint layer ownership (PP) apply.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d1p{pp_size}t{tp_size}",
        pair_name=f"test_megatron_pp{pp_size}tp{tp_size}",
        tag=f"megatron_pp{pp_size}tp{tp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,cp_size",
    [(4, 2), (8, 4)],
    ids=["4gpu-dp1cp2", "8gpu-dp1cp4"],
)
def test_awex_megatron_cp_e2e_weight_update(n_gpus, cp_size, tmp_path_factory):
    """Full round trip: MegatronEngine (pure CP) → weight-update gateway → SGLang.

    CP splits the sequence across ranks for attention but all CP ranks hold
    identical parameters. dp_replicated=True tells awex only one CP rank per
    group needs to send.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d1c{cp_size}",
        pair_name=f"test_megatron_cp{cp_size}",
        tag=f"megatron_cp{cp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,dp_size,cp_size",
    [(8, 2, 2)],
    ids=["8gpu-dp2cp2"],
)
def test_awex_megatron_dp_cp_e2e_weight_update(
    n_gpus, dp_size, cp_size, tmp_path_factory
):
    """Full round trip: MegatronEngine (DP+CP hybrid) → weight-update gateway → SGLang.

    Combines data parallelism with context parallelism. Each DP replica has
    cp_size CP ranks all holding identical parameters. dp_replicated=True
    ensures only one CP rank per DP group sends, avoiding redundant transfers.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{dp_size}c{cp_size}",
        pair_name=f"test_megatron_dp{dp_size}cp{cp_size}",
        tag=f"megatron_dp{dp_size}cp{cp_size}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,ep_size",
    [(4, 2), (8, 4)],
    ids=["4gpu-dp2ep2", "8gpu-dp4ep4"],
)
def test_awex_megatron_ep_e2e_weight_update(n_gpus, ep_size, tmp_path_factory):
    """Full round trip: MegatronEngine (EP) → weight-update gateway → SGLang.

    Each EP rank owns a different subset of expert parameters while attention
    and norm weights are replicated. Uses a truncated Qwen3-30B-A3B MoE model
    (4 layers) with init_from_scratch=True to avoid loading full weights.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    n_train = n_gpus - n_gpus // 2
    tmp = tmp_path_factory.mktemp(f"megatron_ep{ep_size}_model")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{n_train}e{ep_size}",
        pair_name=f"test_megatron_ep{ep_size}",
        tag=f"megatron_ep{ep_size}",
        tmp_path_factory=tmp_path_factory,
        model_path=_make_truncated_moe_model(tmp, num_layers=4),
        validate_param_names=_VALIDATE_PARAM_NAMES_MOE,
        init_from_scratch=True,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize(
    "n_gpus,dp_size,ep_size",
    [(8, 4, 2)],
    ids=["8gpu-dp4ep2"],
)
def test_awex_megatron_dp_ep_e2e_weight_update(
    n_gpus, dp_size, ep_size, tmp_path_factory
):
    """Full round trip: MegatronEngine (DP+EP hybrid) → weight-update gateway → SGLang.

    Combines data parallelism with expert parallelism. Each DP replica has
    ep_size EP ranks owning different expert subsets. Non-expert params
    (attention, norms) are replicated across all ranks.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    tmp = tmp_path_factory.mktemp(f"megatron_dp{dp_size}ep{ep_size}_model")
    _run_megatron_awex_e2e(
        n_gpus=n_gpus,
        backend=f"megatron:d{dp_size}e{ep_size}",
        pair_name=f"test_megatron_dp{dp_size}ep{ep_size}",
        tag=f"megatron_dp{dp_size}ep{ep_size}",
        tmp_path_factory=tmp_path_factory,
        model_path=_make_truncated_moe_model(tmp, num_layers=4),
        validate_param_names=_VALIDATE_PARAM_NAMES_MOE,
        init_from_scratch=True,
    )


# ---------------------------------------------------------------------------
# Colocated weight update: Megatron + SGLang on the SAME GPUs (pure DP)
# ---------------------------------------------------------------------------


def _run_megatron_colocate_e2e(
    *,
    n_gpus: int,
    pair_name: str,
    tag: str,
    tmp_path_factory,
    model_path: str | None = None,
):
    """Colocated weight transfer: MegatronEngine + SGLang share the same GPUs.

    Unlike the separated tests where inference and training each own a
    disjoint half of the GPUs, colocated mode puts both on every GPU.
    The LocalScheduler round-robin counter naturally wraps, giving
    inference GPUs 0..N-1 and training GPUs 0..N-1 (same devices).

    Only pure DP is supported for colocated mode (TP=1, PP=1, EP=1).
    """
    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    tmp = tmp_path_factory.mktemp(tag)
    model_path = model_path or _get_test_model_path()
    scheduler = _make_local_scheduler(tmp, tag, gpu_devices=list(range(n_gpus)))

    # Both inference and training use ALL n_gpus GPUs (colocated).
    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_gpus}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=f"megatron:d{n_gpus}",
        experiment_name=f"test-awex-{tag}",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.megatron_engine.MegatronLMEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None
    try:
        # -- 1. SGLang inference (uses GPUs 0..N-1) -------------------------
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        # Randomize inference weights so the transfer is NOT a no-op.
        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        # -- 2. Megatron training (wraps to same GPUs 0..N-1) ---------------
        train_ctrl.initialize(
            role="actor",
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=2
            ),
        )
        train_worker_urls = list(train_ctrl._worker_addrs)

        # -- 3. Weight update gateway ---------------------------------------
        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(host="127.0.0.1", request_timeout=300.0)
        )
        wu_ctrl.initialize()
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        # -- 4. Connect with colocate=True ----------------------------------
        wu_ctrl.connect(
            pair_name=pair_name,
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            colocate=True,
        )

        # -- 5. Colocated weight update -------------------------------------
        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1
        wu_ctrl.disconnect()

        # -- 6. Verify inference server still works post-update -------------
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

        # -- 7. Validate training ↔ inference parameter equality ------------
        _validate_weight_update_correctness_megatron(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
            tag=tag,
        )
    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_awex_megatron_colocate_dp_e2e_weight_update(n_gpus, tmp_path_factory):
    """Full round trip: colocated MegatronEngine (pure DP) + SGLang on same GPUs.

    Unlike separated tests that split GPUs between training and inference,
    colocated mode shares all GPUs.  Weight transfer uses CUDA IPC
    (zero-copy on same device) instead of NCCL P2P across devices.

    Only pure DP (TP=1, PP=1, EP=1) is supported for colocated mode.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_colocate_e2e(
        n_gpus=n_gpus,
        pair_name=f"test_megatron_colocate_dp{n_gpus}",
        tag=f"megatron_colocate_dp{n_gpus}",
        tmp_path_factory=tmp_path_factory,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
def test_awex_megatron_colocate_dp_multi_version_e2e(tmp_path_factory):
    """Colocated weight update with multiple sequential versions.

    Verifies that the colocated IPC path correctly handles version
    sequencing: version 1 → version 2.  The KV store keys include
    the version number, so each round must produce fresh IPC handles.
    """
    n_gpus = 2
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.v2.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.v2.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    tag = "megatron_colocate_multi_ver"
    tmp = tmp_path_factory.mktemp(tag)
    model_path = _get_test_model_path()
    scheduler = _make_local_scheduler(tmp, tag, gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_gpus}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=f"megatron:d{n_gpus}",
        experiment_name=f"test-awex-{tag}",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.v2.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.megatron_engine.MegatronLMEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None
    try:
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        train_ctrl.initialize(
            role="actor",
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=2
            ),
        )
        train_worker_urls = list(train_ctrl._worker_addrs)

        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(host="127.0.0.1", request_timeout=300.0)
        )
        wu_ctrl.initialize()
        assert wu_ctrl.health_check()

        wu_ctrl.connect(
            pair_name="test_colocate_multi_ver",
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            colocate=True,
        )

        # Version 1
        result1 = wu_ctrl.update_weights(version=1)
        assert result1.status == "ok"
        assert result1.version == 1

        # Version 2
        result2 = wu_ctrl.update_weights(version=2)
        assert result2.status == "ok"
        assert result2.version == 2

        wu_ctrl.disconnect()

        # Verify inference still works after two sequential updates
        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=30.0,
        )
        assert gen_resp.status_code == 200, (
            f"Generation failed after weight updates: {gen_resp.text}"
        )

        # Final parameter equality check
        _validate_weight_update_correctness_megatron(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
            tag=tag,
        )
    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)
