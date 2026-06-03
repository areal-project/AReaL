"""Benchmark weight update latency across Awex (nccl/file) and XCCL."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
from contextlib import contextmanager

import torch

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from areal.api.alloc_mode import ModelAllocation, ParallelStrategy
from areal.api.cli_args import (
    InferenceEngineConfig,
    MegatronEngineConfig,
    TrainEngineConfig,
    vLLMConfig,
)
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.engine.megatron_engine import MegatronEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.utils import network

DEFAULT_DENSE_PATH = "/home/model/Qwen3-0.6B"
DEFAULT_MOE_PATH = "/home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l2-e8"


def _detect_device_backend(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    if getattr(torch, "npu", None) is not None and torch.npu.is_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _visible_env_name(device_backend: str) -> str:
    if device_backend == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"


def _device_count(device_backend: str) -> int:
    if device_backend == "npu":
        return 0 if getattr(torch, "npu", None) is None else torch.npu.device_count()
    if device_backend == "cuda":
        return torch.cuda.device_count()
    return 0


def _select_devices(tp_size: int = 1, device_backend: str = "cuda"):
    visible_env = os.environ.get(_visible_env_name(device_backend), "").strip()
    if visible_env:
        visible_devices = [int(x) for x in visible_env.split(",") if x.strip()]
    else:
        visible_devices = list(range(_device_count(device_backend)))

    need = 1 + tp_size
    if len(visible_devices) < need:
        raise RuntimeError(
            f"Need at least {need} devices (1 for Megatron + {tp_size} for vLLM). "
            f"Found {len(visible_devices)}."
        )

    megatron_device = visible_devices[0]
    vllm_devices = visible_devices[1 : 1 + tp_size]
    return megatron_device, vllm_devices


@contextmanager
def _temp_env(overrides: dict[str, str]):
    old = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_model_path(model_kind: str) -> str:
    if model_kind == "dense":
        return (
            os.environ.get("AREAL_AWEX_DENSE_MODEL_PATH")
            or os.environ.get("AREAL_BENCH_DENSE_MODEL_PATH")
            or DEFAULT_DENSE_PATH
        )
    return (
        os.environ.get("AREAL_AWEX_MOE_MODEL_PATH")
        or os.environ.get("AREAL_BENCH_MOE_MODEL_PATH")
        or DEFAULT_MOE_PATH
    )


def _build_vllm_args(model_path: str) -> dict:
    vllm_cfg = vLLMConfig(
        skip_tokenizer_init=False,
        model=model_path,
        gpu_memory_utilization=0.7,
        max_num_seqs=1,
        max_model_len=2048,
        enforce_eager=True,
    )
    if hasattr(vllm_cfg, "load_format"):
        vllm_cfg.load_format = os.environ.get(
            "AREAL_AWEX_VLLM_LOAD_FORMAT",
            os.environ.get("AREAL_BENCH_VLLM_LOAD_FORMAT", "auto"),
        )
    return vLLMConfig.build_args(vllm_cfg, tp_size=1, pp_size=1)


def _setup_megatron_env(device_backend: str):
    dist_port = network.find_free_ports(1)[0]
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(dist_port),
        }
    )
    if device_backend == "npu":
        if getattr(torch, "npu", None) is None:
            raise RuntimeError("torch.npu is not available for NPU backend.")
        torch.npu.set_device(0)
    else:
        torch.cuda.set_device(0)


def _safe_destroy(fn, name: str, timeout: int = 10) -> None:
    done = threading.Event()

    def _run():
        try:
            fn()
        finally:
            done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    if not done.wait(timeout):
        print(f"[bench] {name} timed out after {timeout}s; continuing.", flush=True)


def _run_mode(
    mode: str,
    model_kind: str,
    iters: int,
    warmup: int,
    device_backend: str,
) -> dict:
    from awex.meta.meta_server import start_meta_server, stop_meta_server

    megatron_device, vllm_devices = _select_devices(
        tp_size=1, device_backend=device_backend
    )
    model_path = _resolve_model_path(model_kind)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    vllm_env = {
        _visible_env_name(device_backend): ",".join(map(str, vllm_devices)),
        "AWEX_DEVICE_TYPE": device_backend,
    }
    # NOTE: Do NOT pin VLLM_PLUGINS to "awex_adapter". The awex vLLM plugin is
    # auto-discovered from the installed package, and restricting VLLM_PLUGINS
    # would also exclude the vllm_ascend platform plugin, breaking NPU device
    # detection ("Failed to infer device type").

    inf_engine = None
    train_engine = None
    meta_server_addr = None
    tmpdir = None

    timings: list[float] = []

    try:
        if mode.startswith("awex"):
            meta_ip, meta_port = start_meta_server()
            meta_server_addr = f"{meta_ip}:{meta_port}"

        vllm_args = _build_vllm_args(model_path)
        inf_engine = RemotevLLMEngine(
            InferenceEngineConfig(
                experiment_name="awex_bench",
                trial_name=f"{mode}_{model_kind}",
                setup_timeout=360,
                request_timeout=60,
            )
        )
        with _temp_env(vllm_env):
            inf_engine.launch_server(vllm_args)
        # Single vLLM instance (tp=1, dp=1). Pass DP size explicitly because the
        # bench builds InferenceEngineConfig without a `backend` string, so
        # initialize() cannot derive it from self.config.backend.
        inf_engine.initialize(train_data_parallel_size=1)
        inf_engine.set_version(1)

        _setup_megatron_env(device_backend)

        train_config = TrainEngineConfig(
            experiment_name="awex_bench",
            trial_name=f"{mode}_{model_kind}",
            path=model_path,
            init_from_scratch=False,
            optimizer=None,
            megatron=MegatronEngineConfig(),
        )
        train_engine = MegatronEngine(train_config)
        # New per-engine allocation API (AllocationMode was removed): build the
        # train process group from a ParallelStrategy and the rollout/gen
        # allocation from a ModelAllocation backend string.
        train_parallel = ParallelStrategy(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        )
        gen_allocation = ModelAllocation.from_str("vllm:d1p1t1", name="rollout")
        train_engine.create_process_group(train_parallel)
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=16, train_batch_size=2
        )
        train_engine.initialize(addr=None, ft_spec=ft_spec)
        train_engine.set_version(1)

        if mode == "xccl":
            meta = WeightUpdateMeta.from_megatron_xccl(gen_allocation=gen_allocation)
        elif mode == "awex_nccl":
            if meta_server_addr is None:
                raise RuntimeError("Meta server addr missing for awex_nccl")
            meta = WeightUpdateMeta.from_awex(
                meta_server_addr=meta_server_addr,
                comm_backend="hccl" if device_backend == "npu" else "nccl",
            )
        elif mode == "awex_file":
            if meta_server_addr is None:
                raise RuntimeError("Meta server addr missing for awex_file")
            tmpdir = tempfile.TemporaryDirectory()
            meta = WeightUpdateMeta.from_awex(
                meta_server_addr=meta_server_addr,
                comm_backend="file",
            )
            meta.path = tmpdir.name
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if device_backend == "npu" and meta.type == "awex":
            meta.weights_exchange_ipc_backend = "cpu"

        train_engine.connect_engine(inf_engine, meta)

        for i in range(warmup):
            train_engine.set_version(i + 1)
            train_engine.update_weights(meta)

        for i in range(iters):
            train_engine.set_version(warmup + i + 1)
            start = time.perf_counter()
            train_engine.update_weights(meta)
            timings.append(time.perf_counter() - start)

        print(f"[bench] {mode} updates complete, entering cleanup.", flush=True)
    finally:
        print(f"[bench] cleanup start for {mode}.", flush=True)
        if train_engine is not None:
            print(f"[bench] destroying train engine for {mode}.", flush=True)
            _safe_destroy(train_engine.destroy, "train_engine.destroy", timeout=10)
        if inf_engine is not None:
            print(f"[bench] destroying inference engine for {mode}.", flush=True)
            _safe_destroy(inf_engine.destroy, "inf_engine.destroy", timeout=10)
        if tmpdir is not None:
            tmpdir.cleanup()
        if mode.startswith("awex"):
            print(f"[bench] stopping meta server for {mode}.", flush=True)
            stop_meta_server()
        print(f"[bench] cleanup done for {mode}.", flush=True)

    return {
        "mode": mode,
        "iterations": iters,
        "warmup": warmup,
        "timings_sec": timings,
        "mean_sec": statistics.mean(timings) if timings else None,
        "p50_sec": statistics.median(timings) if timings else None,
        "min_sec": min(timings) if timings else None,
        "max_sec": max(timings) if timings else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        default="awex_nccl,awex_file,xccl",
        help="Comma-separated modes: awex_nccl, awex_file, xccl",
    )
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--model-kind", choices=["dense", "moe"], default="dense")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--device-backend",
        choices=["auto", "cuda", "npu", "cpu"],
        default="auto",
        help="Device backend to use (auto/cuda/npu/cpu).",
    )
    args = parser.parse_args()
    device_backend = _detect_device_backend(args.device_backend)
    if device_backend != "auto":
        os.environ["AWEX_DEVICE_TYPE"] = device_backend
        if device_backend == "npu":
            os.environ.setdefault("AWEX_USE_MINDSPEED", "1")

    results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        try:
            res = _run_mode(
                mode,
                args.model_kind,
                args.iters,
                args.warmup,
                device_backend,
            )
            results.append(res)
        except Exception as exc:  # pylint: disable=broad-except
            results.append({"mode": mode, "error": repr(exc)})

    payload = {
        "model_kind": args.model_kind,
        "iters": args.iters,
        "warmup": args.warmup,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
