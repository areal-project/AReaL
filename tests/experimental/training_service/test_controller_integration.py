from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import requests
import torch

from areal.api.cli_args import (
    MicroBatchSpec,
    OptimizerConfig,
    SchedulingSpec,
    TrainEngineConfig,
)
from areal.api.io_struct import FinetuneSpec, SaveLoadMeta
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.infra.platforms import current_platform
from areal.infra.scheduler.local import LocalScheduler

LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
LOCAL_MOE_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-30B-A3B/"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _resolve_model_path_or_skip() -> str:
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH
    pytest.skip(
        "Local model path not found for CUDA integration test: "
        f"{LOCAL_MODEL_PATH} (HF model: Qwen/Qwen3-0.6B)"
    )
    raise RuntimeError("unreachable after pytest.skip")


def _resolve_moe_model_path_or_skip() -> str:
    if os.path.exists(LOCAL_MOE_MODEL_PATH):
        return LOCAL_MOE_MODEL_PATH
    pytest.skip(
        "Local MoE model path not found for CUDA integration test: "
        f"{LOCAL_MOE_MODEL_PATH} (HF model: Qwen/Qwen3-30B-A3B)"
    )
    raise RuntimeError("unreachable after pytest.skip")


@dataclass(frozen=True)
class _StrategyCase:
    name: str
    train_engine: str
    backend: str
    model_resolver: Callable[[], str]
    expected_dp_size: int


def _strategy_cases_2gpu() -> list[_StrategyCase]:
    return [
        _StrategyCase(
            name="fsdp_dp2",
            train_engine="areal.engine.FSDPEngine",
            backend="fsdp:d2",
            model_resolver=_resolve_model_path_or_skip,
            expected_dp_size=2,
        ),
        _StrategyCase(
            name="megatron_dp2",
            train_engine="areal.engine.MegatronEngine",
            backend="megatron:d2",
            model_resolver=_resolve_model_path_or_skip,
            expected_dp_size=2,
        ),
        _StrategyCase(
            name="megatron_tp2",
            train_engine="areal.engine.MegatronEngine",
            backend="megatron:t2",
            model_resolver=_resolve_model_path_or_skip,
            expected_dp_size=1,
        ),
        _StrategyCase(
            name="megatron_cp2",
            train_engine="areal.engine.MegatronEngine",
            backend="megatron:c2",
            model_resolver=_resolve_model_path_or_skip,
            expected_dp_size=1,
        ),
        _StrategyCase(
            name="megatron_pp2",
            train_engine="areal.engine.MegatronEngine",
            backend="megatron:p2",
            model_resolver=_resolve_model_path_or_skip,
            expected_dp_size=1,
        ),
        _StrategyCase(
            name="megatron_ep2",
            train_engine="areal.engine.MegatronEngine",
            backend="megatron:d2e2",
            model_resolver=_resolve_moe_model_path_or_skip,
            expected_dp_size=2,
        ),
    ]


@contextmanager
def _build_cuda_gateway_controller(
    tmp_path_factory: pytest.TempPathFactory,
    *,
    case: _StrategyCase,
):
    if current_platform.device_count() < 2:
        pytest.skip("This test requires 2 GPUs")

    model_path = case.model_resolver()
    tmp_path = tmp_path_factory.mktemp(f"training_gateway_cuda_{case.name}")
    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()

    scheduler = LocalScheduler(
        gpu_devices=[0, 1],
        log_dir=str(tmp_path),
        enable_tms_offload=True,
        experiment_name=f"test_training_gateway_controller_cuda_{case.name}",
        trial_name="trial_0",
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )

    config = TrainEngineConfig(
        experiment_name=f"test_training_gateway_controller_cuda_{case.name}",
        trial_name="trial_0",
        backend=case.backend,
        scheduling_spec=(
            SchedulingSpec(
                cpu=1,
                gpu=1,
                mem=2048,
                port_count=1,
                cmd="python -m areal.infra.rpc.rpc_server",
            ),
        ),
        path=model_path,
        admin_api_key=f"test-admin-key-cuda-{case.name}",
        request_timeout=180.0,
        setup_timeout=300.0,
        offload=False,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=128),
        optimizer=OptimizerConfig(),
    )

    controller = GatewayTrainController(
        train_engine=case.train_engine,
        scheduler=scheduler,
        config=config,
    )
    try:
        controller.initialize(
            role=f"train-gateway-cuda-{case.name}",
            ft_spec=FinetuneSpec(
                total_train_epochs=1,
                dataset_size=8,
                train_batch_size=2,
            ),
            wait=True,
        )
    except Exception as exc:
        try:
            controller.destroy()
        finally:
            scheduler.delete_workers(role=None)
        msg = str(exc).lower()
        if "out of memory" in msg:
            pytest.skip(
                f"Skipping {case.name} due to transient CUDA OOM during bootstrap: {exc}"
            )
        raise

    try:
        yield controller, tmp_path_factory
    finally:
        controller.destroy()
        scheduler.delete_workers(role=None)


def _make_batch(n: int = 4, seq_len: int = 16) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "input_ids": torch.randint(0, 100, (1, seq_len), dtype=torch.long),
            "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
            "loss_mask": torch.ones((1, seq_len), dtype=torch.bool),
        }
        for _ in range(n)
    ]


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.parametrize("case", _strategy_cases_2gpu(), ids=lambda c: c.name)
def test_gateway_controller_integration(
    tmp_path_factory: pytest.TempPathFactory,
    case: _StrategyCase,
):
    with _build_cuda_gateway_controller(tmp_path_factory, case=case) as (
        controller,
        tmp_factory,
    ):
        # -- health --------------------------------------------------------

        gateway_resp = requests.get(f"{controller._gateway_addr}/health", timeout=15)
        assert gateway_resp.status_code == 200
        assert gateway_resp.json()["status"] == "ok"

        router_resp = requests.get(f"{controller._router_addr}/health", timeout=15)
        assert router_resp.status_code == 200
        assert router_resp.json()["status"] == "ok"

        # -- topology ------------------------------------------------------

        topology_resp = requests.get(f"{controller._model_addr}/topology", timeout=15)
        assert topology_resp.status_code == 200
        topology = topology_resp.json()

        assert topology["dp_size"] == case.expected_dp_size
        assert len(topology["workers"]) == 2
        worker_dp_ranks = sorted(w["dp_rank"] for w in topology["workers"])
        worker_dp_heads = [w["is_dp_head"] for w in topology["workers"]]

        if case.expected_dp_size == 2:
            assert len(topology["dp_heads"]) == 2
            assert len(topology["dp_groups"]) == 2
            assert all(len(g) == 1 for g in topology["dp_groups"])
            assert worker_dp_ranks == [0, 1]
            assert worker_dp_heads.count(True) == 2
        else:
            assert len(topology["dp_heads"]) == 1
            assert len(topology["dp_groups"]) == 1
            assert len(topology["dp_groups"][0]) == 2
            assert worker_dp_ranks == [0, 0]
            assert worker_dp_heads.count(True) == 1

        # -- train / eval mode toggle --------------------------------------

        controller.train(mode=False)
        controller.train(mode=True)
        controller.eval()

        # -- version -------------------------------------------------------

        controller.set_version(11)
        assert controller.get_version() == 11

        controller.set_version(23)
        assert controller.get_version() == 23

        # -- forward_batch -------------------------------------------------

        forward_result = controller.forward_batch(_make_batch(4))
        assert forward_result is not None

        # -- export_stats --------------------------------------------------

        stats = controller.export_stats()
        assert isinstance(stats, dict)

        # -- offload / onload cycle ----------------------------------------

        controller.offload()
        controller.onload()

        # -- save / load ---------------------------------------------------

        model_path = case.model_resolver()
        save_load_path = str(tmp_factory.mktemp("hf_saveload"))
        save_meta = SaveLoadMeta(
            path=save_load_path,
            weight_format="hf",
            with_optim=False,
            base_model_path=model_path,
        )
        controller.save(save_meta)

        load_meta = SaveLoadMeta(
            path=save_load_path,
            weight_format="hf",
            with_optim=False,
            base_model_path=model_path,
        )
        controller.load(load_meta)

        # -- step_lr_scheduler ---------------------------------------------

        controller.step_lr_scheduler()

        # -- optimizer_zero_grad / optimizer_step --------------------------

        controller.optimizer_zero_grad()
        controller.optimizer_step()

        # -- clear_batches -------------------------------------------------

        controller.clear_batches()

        # -- second offload / onload cycle ---------------------------------

        controller.offload()
        controller.onload()

        # -- final stats ---------------------------------------------------

        stats = controller.export_stats()
        assert isinstance(stats, dict)
