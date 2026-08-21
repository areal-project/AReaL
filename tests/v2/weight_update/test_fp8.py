# SPDX-License-Identifier: Apache-2.0

import os

import httpx
import pytest
import torch

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

_VALIDATE_PARAM_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.weight_scale_inv",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.k_proj.weight_scale_inv",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.v_proj.weight_scale_inv",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.gate_proj.weight_scale_inv",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.up_proj.weight_scale_inv",
    "model.layers.27.self_attn.q_proj.weight",
    "model.layers.27.self_attn.q_proj.weight_scale_inv",
    "model.norm.weight",
]


def _get_test_model_path() -> str:
    local = "/storage/openpsi/models/Qwen__Qwen3-0.6B-FP8/"
    if os.path.isdir(local):
        return local
    return "Qwen/Qwen3-0.6B-FP8"


def _make_local_scheduler(tmp_path, name: str, gpu_devices: list[int]):
    from areal.infra.scheduler.local import LocalScheduler

    fileroot = tmp_path / f"{name}_fileroot"
    fileroot.mkdir(exist_ok=True)
    nr_root = tmp_path / f"{name}_name_resolve"
    nr_root.mkdir(exist_ok=True)

    return LocalScheduler(
        gpu_devices=gpu_devices,
        log_dir=str(tmp_path / f"{name}_logs"),
        experiment_name=f"test-fp8-{name}",
        trial_name="t0",
        fileroot=str(fileroot),
        nfs_record_root=str(nr_root),
    )


def _validate_weight_update_correctness(
    train_worker_urls: list[str],
    inf_worker_urls: list[str],
    param_dir,
) -> None:
    print(
        "\n[weight-validation] Fetching parameters from 1 training "
        "worker and 1 inference worker …"
    )

    train_worker_url = train_worker_urls[0]
    inf_worker_url = inf_worker_urls[0]

    train_path = str(param_dir / "train_params.pt")
    resp = httpx.post(
        f"{train_worker_url}/awex/debug/get_parameters",
        json={"save_path": train_path, "names": _VALIDATE_PARAM_NAMES},
        timeout=120.0,
    )
    assert resp.status_code == 200, (
        f"get_parameters failed on training worker: {resp.text}"
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

    train_params = torch.load(train_path, map_location="cpu", weights_only=True)
    infer_params = torch.load(inf_path, map_location="cpu", weights_only=True)

    print(f"[weight-validation] Comparing {len(_VALIDATE_PARAM_NAMES)} parameters …")
    for name in _VALIDATE_PARAM_NAMES:
        assert name in train_params, f"Training missing: {name}"
        assert name in infer_params, f"Inference missing: {name}"

        train_tensor = train_params[name]
        infer_tensor = infer_params[name]

        if train_tensor.dtype == torch.float8_e4m3fn:
            assert infer_tensor.dtype == torch.float8_e4m3fn, (
                f"Param '{name}' was dequantized from float8_e4m3fn"
                f" to {infer_tensor.dtype} during transfer"
            )

        torch.testing.assert_close(
            train_tensor,
            infer_tensor,
            rtol=0,
            atol=0,
            msg=f"Parameter mismatch after weight update: {name}",
        )
        print(
            f"[weight-validation]   {name}: OK "
            f"(shape={list(train_tensor.shape)}, dtype={train_tensor.dtype})"
        )

    print(
        f"[weight-validation] All {len(_VALIDATE_PARAM_NAMES)} parameters "
        f"match between Megatron FP8 training and inference ✓"
    )


def _run_megatron_fp8_awex_e2e(
    *,
    n_gpus: int,
    pair_name: str,
    tag: str,
    tmp_path_factory,
    colocate: bool,
    model_path: str | None = None,
):
    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        FP8EngineConfig,
        InferenceEngineConfig,
        MegatronEngineConfig,
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

    if colocate:
        dp = n_gpus
    else:
        dp = n_gpus // 2

    tmp = tmp_path_factory.mktemp(tag)
    model_path = model_path or _get_test_model_path()
    scheduler = _make_local_scheduler(tmp, tag, gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{dp}",
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
        backend=f"megatron:d{dp}",
        experiment_name=f"test-fp8-awex-{tag}",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        megatron=MegatronEngineConfig(
            fp8_config=FP8EngineConfig(
                mode="e4m3",
                recipe="blockwise",
                param=True,
                direct_convert=True,
            ),
        ),
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
        # -- 1. SGLang inference ---------------------------------------------
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        # Randomize inference weights so the transfer is NOT a no-op.
        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        # -- 2. Megatron training --------------------------------------------
        train_ctrl.initialize(
            role="actor",
            ft_spec=FinetuneSpec(
                total_train_epochs=1, dataset_size=100, train_batch_size=2
            ),
            wait=True,
        )
        train_worker_urls = list(train_ctrl._worker_addrs)

        # -- 3. Weight update gateway ----------------------------------------
        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(host="127.0.0.1", request_timeout=300.0)
        )
        wu_ctrl.initialize()
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        # -- 4. Connect with colocate=True/False -----------------------------
        wu_ctrl.connect(
            pair_name=pair_name,
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            colocate=colocate,
            nccl_master_addr="127.0.0.1",
            nccl_master_port=find_free_ports(1)[0],
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
        _validate_weight_update_correctness(
            train_worker_urls=train_worker_urls,
            inf_worker_urls=inf_worker_urls,
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
def test_awex_megatron_fp8_xccl_dp_e2e_weight_update(n_gpus, tmp_path_factory):
    """Full round trip: MegatronEngine (pure DP) + SGLang for FP8 model on
    separated GPUs.

    Weight transfer uses NCCL P2P across devices.

    Only pure DP (TP=1, PP=1, EP=1) is supported.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_fp8_awex_e2e(
        n_gpus=n_gpus,
        pair_name=f"test_fp8_xccl_dp{n_gpus}",
        tag=f"xccl_dp{n_gpus}",
        tmp_path_factory=tmp_path_factory,
        colocate=False,
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_awex_megatron_fp8_colocate_dp_e2e_weight_update(n_gpus, tmp_path_factory):
    """Full round trip: colocated MegatronEngine (pure DP) + SGLang for FP8 model
    on same GPUs.

    Unlike separated tests that split GPUs between training and inference,
    colocated mode shares all GPUs.  Weight transfer uses CUDA IPC
    (zero-copy on same device) instead of NCCL P2P across devices.

    Only pure DP (TP=1, PP=1, EP=1) is supported.
    """

    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")
    _run_megatron_fp8_awex_e2e(
        n_gpus=n_gpus,
        pair_name=f"test_fp8_colocate_dp{n_gpus}",
        tag=f"colocate_dp{n_gpus}",
        tmp_path_factory=tmp_path_factory,
        colocate=True,
    )
