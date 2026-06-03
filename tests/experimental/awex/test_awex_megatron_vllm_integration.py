"""Integration test for Awex weight exchange between Megatron and vLLM."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist

from tests.utils import get_model_path

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import (
    InferenceEngineConfig,
    MegatronEngineConfig,
    TrainEngineConfig,
    vLLMConfig,
)
from areal.api.io_struct import FinetuneSpec, WeightUpdateMeta
from areal.engine.megatron_engine import MegatronEngine
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.infra.utils.proc import kill_process_tree
from areal.utils import network
from areal.utils.pkg_version import is_available

IS_VLLM_INSTALLED = is_available("vllm")
IS_AWEX_INSTALLED = is_available("awex")

DENSE_HF_ID = "Qwen/Qwen3-0.6B"
DENSE_LOCAL_PATH = "/home/model/Qwen3-0.6B/"
MOE_LOCAL_PATH = "/home/model/Qwen3-30B-A3B-Instruct-2507-reduced-l2-e8"

# Avoid MindSpeed transformer_config_init_wrapper get sys args which might translate to ''
sys.argv = [sys.argv[0]]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_csv(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


# === Parallel configuration (edit to test multi-GPU) ===
# Training-side parallelism (Megatron).
TRAIN_TP_SIZE = _env_int("AREAL_AWEX_TRAIN_TP_SIZE", 1)
TRAIN_PP_SIZE = _env_int("AREAL_AWEX_TRAIN_PP_SIZE", 1)
TRAIN_DP_SIZE = _env_int("AREAL_AWEX_TRAIN_DP_SIZE", 1)
TRAIN_CP_SIZE = _env_int("AREAL_AWEX_TRAIN_CP_SIZE", 1)
TRAIN_EP_SIZE = _env_int("AREAL_AWEX_TRAIN_EP_SIZE", 1)
TRAIN_ETP_SIZE = _env_int("AREAL_AWEX_TRAIN_ETP_SIZE", 1)

# Inference-side parallelism (vLLM).
VLLM_TP_SIZE = _env_int("AREAL_AWEX_VLLM_TP_SIZE", 1)
VLLM_PP_SIZE = _env_int("AREAL_AWEX_VLLM_PP_SIZE", 1)
# Optional vLLM data/expert parallel settings (internal DP on a single server).
VLLM_DATA_PARALLEL_SIZE = _env_int("AREAL_AWEX_VLLM_DP_SIZE", 1)
VLLM_ENABLE_EXPERT_PARALLEL = _env_bool("AREAL_AWEX_VLLM_ENABLE_EP", False)
VLLM_ENABLE_EPLB = _env_bool("AREAL_AWEX_VLLM_ENABLE_EPLB", False)
VLLM_EXPERT_PLACEMENT_STRATEGY = os.environ.get(
    "AREAL_AWEX_VLLM_EXPERT_PLACEMENT_STRATEGY", ""
).strip()
VLLM_EPLB_WINDOW_SIZE = _env_int("AREAL_AWEX_VLLM_EPLB_WINDOW_SIZE", 8)
VLLM_EPLB_STEP_INTERVAL = _env_int("AREAL_AWEX_VLLM_EPLB_STEP_INTERVAL", 4)
VLLM_EPLB_NUM_REDUNDANT_EXPERTS = _env_int(
    "AREAL_AWEX_VLLM_EPLB_NUM_REDUNDANT_EXPERTS", 0
)
VLLM_EPLB_LOG_BALANCEDNESS = _env_bool("AREAL_AWEX_VLLM_EPLB_LOG_BALANCEDNESS", True)
# NPU(vllm-ascend) EPLB knobs. Keep GPU-style knobs above for CUDA vLLM.
VLLM_EPLB_DYNAMIC = _env_bool("AREAL_AWEX_VLLM_EPLB_DYNAMIC", True)
VLLM_EPLB_EXPERT_HEAT_COLLECTION_INTERVAL = _env_int(
    "AREAL_AWEX_VLLM_EPLB_EXPERT_HEAT_COLLECTION_INTERVAL",
    VLLM_EPLB_WINDOW_SIZE,
)
VLLM_EPLB_ALGORITHM_EXECUTION_INTERVAL = _env_int(
    "AREAL_AWEX_VLLM_EPLB_ALGORITHM_EXECUTION_INTERVAL",
    VLLM_EPLB_STEP_INTERVAL,
)
VLLM_EPLB_EXPERT_MAP_PATH = (
    os.environ.get("AREAL_AWEX_VLLM_EPLB_EXPERT_MAP_PATH", "").strip() or None
)
VLLM_EPLB_EXPERT_MAP_RECORD_PATH = (
    os.environ.get("AREAL_AWEX_VLLM_EPLB_EXPERT_MAP_RECORD_PATH", "").strip() or None
)
VLLM_EPLB_POLICY_TYPE = _env_int("AREAL_AWEX_VLLM_EPLB_POLICY_TYPE", 1)
VLLM_NUM_ENGINES = _env_int("AREAL_AWEX_VLLM_NUM_ENGINES", 1)
NUM_UPDATES = _env_int("AREAL_AWEX_NUM_UPDATES", 1)
VALIDATION_STEPS = _env_int("AREAL_AWEX_VALIDATION_STEPS", -1)
VALIDATE_EVERY_N_STEPS = _env_int("AREAL_AWEX_VALIDATE_EVERY_N_STEPS", 1)
REQUESTS_PER_UPDATE = _env_int("AREAL_AWEX_REQUESTS_PER_UPDATE", 0)
REQUEST_TIMEOUT_S = _env_int("AREAL_AWEX_REQUEST_TIMEOUT_S", 30)
REQUEST_RETRIES = _env_int("AREAL_AWEX_REQUEST_RETRIES", 3)
STRICT_REQUEST_TRAFFIC = _env_bool("AREAL_AWEX_STRICT_REQUEST_TRAFFIC", False)
ENGINE_REQUEST_TIMEOUT_S = _env_int("AREAL_AWEX_ENGINE_REQUEST_TIMEOUT_S", 60)
ENGINE_SETUP_TIMEOUT_S = _env_int("AREAL_AWEX_ENGINE_SETUP_TIMEOUT_S", 360)
VLLM_GPU_MEMORY_UTILIZATION = _env_float("AREAL_AWEX_VLLM_GPU_MEMORY_UTILIZATION", 0.6)
VLLM_MAX_NUM_SEQS = _env_int("AREAL_AWEX_VLLM_MAX_NUM_SEQS", 1)
VLLM_MAX_MODEL_LEN = _env_int("AREAL_AWEX_VLLM_MAX_MODEL_LEN", 128)
TRAIN_CLUSTER_ID = os.environ.get("AREAL_AWEX_TRAIN_CLUSTER_ID", "").strip() or None
VLLM_CLUSTER_IDS = _env_csv("AREAL_AWEX_VLLM_CLUSTER_IDS")

# vLLM instances to launch (edit for multi-engine setups).
# Each instance can define its own tp/pp sizes and device list (IDs in visible list).
# For internal vLLM DP+EP, set data_parallel_size on the instance. We keep
# a single server instance and let vLLM manage DP ranks internally.
# Optional "engine_rank" controls ordering (default: list order). If provided, it
# must be unique and contiguous starting from 0.
# If all devices are None, they will be auto-assigned after training devices.
if VLLM_NUM_ENGINES < 1:
    raise RuntimeError("AREAL_AWEX_VLLM_NUM_ENGINES must be >= 1.")
if VLLM_CLUSTER_IDS and len(VLLM_CLUSTER_IDS) != VLLM_NUM_ENGINES:
    raise RuntimeError(
        "AREAL_AWEX_VLLM_CLUSTER_IDS must have exactly "
        f"{VLLM_NUM_ENGINES} items, got {len(VLLM_CLUSTER_IDS)}."
    )

VLLM_INSTANCES: list[dict] = []
for engine_idx in range(VLLM_NUM_ENGINES):
    VLLM_INSTANCES.append(
        {
            "tp_size": VLLM_TP_SIZE,
            "pp_size": VLLM_PP_SIZE,
            "devices": None,
            "engine_rank": engine_idx,
            "data_parallel_size": VLLM_DATA_PARALLEL_SIZE,
            "enable_expert_parallel": VLLM_ENABLE_EXPERT_PARALLEL,
            "enable_eplb": VLLM_ENABLE_EPLB,
            "expert_placement_strategy": VLLM_EXPERT_PLACEMENT_STRATEGY or None,
            "awex_cluster_id": (
                VLLM_CLUSTER_IDS[engine_idx] if VLLM_CLUSTER_IDS else None
            ),
            "eplb_config": {
                "window_size": VLLM_EPLB_WINDOW_SIZE,
                "step_interval": VLLM_EPLB_STEP_INTERVAL,
                "num_redundant_experts": VLLM_EPLB_NUM_REDUNDANT_EXPERTS,
                "log_balancedness": VLLM_EPLB_LOG_BALANCEDNESS,
            },
        }
    )


def _detect_device_backend() -> str:
    requested = os.environ.get("AREAL_AWEX_DEVICE_BACKEND", "auto").lower()
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


def _select_devices(
    train_world_size: int,
    vllm_world_size: int,
    device_backend: str = "cuda",
    vllm_device_ids: list[int] | None = None,
):
    visible_env = os.environ.get(_visible_env_name(device_backend), "").strip()
    if visible_env:
        visible_devices = [int(x) for x in visible_env.split(",") if x.strip()]
    else:
        visible_devices = list(range(_device_count(device_backend)))

    if vllm_device_ids is not None:
        for device_id in vllm_device_ids:
            if device_id not in visible_devices:
                raise RuntimeError(
                    f"vLLM device id {device_id} is not visible. "
                    f"Visible devices: {visible_devices}"
                )
        vllm_devices = vllm_device_ids
        train_devices = [d for d in visible_devices if d not in vllm_devices][
            :train_world_size
        ]
    else:
        train_devices = visible_devices[:train_world_size]
        vllm_devices = visible_devices[
            train_world_size : train_world_size + vllm_world_size
        ]

    need = train_world_size + vllm_world_size
    if (
        len(visible_devices) < need
        or len(train_devices) < train_world_size
        or len(vllm_devices) < vllm_world_size
    ):
        pytest.skip(
            f"Need at least {need} devices (train={train_world_size}, vLLM={vllm_world_size}). "
            f"Found {len(visible_devices)}."
        )

    return train_devices, vllm_devices


def _train_world_size() -> int:
    return TRAIN_TP_SIZE * TRAIN_PP_SIZE * TRAIN_DP_SIZE * TRAIN_CP_SIZE


def _build_ascend_eplb_config(base_config: dict | None) -> dict:
    cfg = {
        "dynamic_eplb": VLLM_EPLB_DYNAMIC,
        "expert_heat_collection_interval": VLLM_EPLB_EXPERT_HEAT_COLLECTION_INTERVAL,
        "algorithm_execution_interval": VLLM_EPLB_ALGORITHM_EXECUTION_INTERVAL,
        "num_redundant_experts": VLLM_EPLB_NUM_REDUNDANT_EXPERTS,
        "eplb_policy_type": VLLM_EPLB_POLICY_TYPE,
    }
    if VLLM_EPLB_EXPERT_MAP_PATH:
        cfg["expert_map_path"] = VLLM_EPLB_EXPERT_MAP_PATH
    if VLLM_EPLB_EXPERT_MAP_RECORD_PATH:
        cfg["expert_map_record_path"] = VLLM_EPLB_EXPERT_MAP_RECORD_PATH
    raw = base_config or {}
    if "dynamic_eplb" in raw:
        cfg["dynamic_eplb"] = bool(raw["dynamic_eplb"])
    if "expert_heat_collection_interval" in raw:
        cfg["expert_heat_collection_interval"] = int(
            raw["expert_heat_collection_interval"]
        )
    if "algorithm_execution_interval" in raw:
        cfg["algorithm_execution_interval"] = int(raw["algorithm_execution_interval"])
    # Backward-compatible mapping from GPU EPLB keys.
    if "window_size" in raw and "expert_heat_collection_interval" not in raw:
        cfg["expert_heat_collection_interval"] = int(raw["window_size"])
    if "step_interval" in raw and "algorithm_execution_interval" not in raw:
        cfg["algorithm_execution_interval"] = int(raw["step_interval"])
    if "num_redundant_experts" in raw:
        cfg["num_redundant_experts"] = int(raw["num_redundant_experts"])
    if "expert_map_path" in raw and raw["expert_map_path"]:
        cfg["expert_map_path"] = str(raw["expert_map_path"])
    if "expert_map_record_path" in raw and raw["expert_map_record_path"]:
        cfg["expert_map_record_path"] = str(raw["expert_map_record_path"])
    if "eplb_policy_type" in raw:
        cfg["eplb_policy_type"] = int(raw["eplb_policy_type"])
    return cfg


def _build_vllm_instances() -> list[dict]:
    if not VLLM_INSTANCES:
        raise RuntimeError(
            "VLLM_INSTANCES is empty; configure at least one vLLM instance."
        )
    instances = []
    for idx, inst in enumerate(VLLM_INSTANCES):
        tp_size = int(inst.get("tp_size", VLLM_TP_SIZE))
        pp_size = int(inst.get("pp_size", VLLM_PP_SIZE))
        devices = inst.get("devices")
        engine_rank = int(inst.get("engine_rank", idx))
        data_parallel_size = int(
            inst.get("data_parallel_size", VLLM_DATA_PARALLEL_SIZE) or 1
        )
        enable_expert_parallel = bool(
            inst.get("enable_expert_parallel", VLLM_ENABLE_EXPERT_PARALLEL)
        )
        enable_eplb = bool(inst.get("enable_eplb", VLLM_ENABLE_EPLB))
        expert_placement_strategy = (
            inst.get("expert_placement_strategy") or VLLM_EXPERT_PLACEMENT_STRATEGY
        ) or None
        awex_cluster_id = str(inst.get("awex_cluster_id", "") or "").strip() or None
        eplb_config = inst.get("eplb_config")
        if eplb_config is None:
            eplb_config = {
                "window_size": VLLM_EPLB_WINDOW_SIZE,
                "step_interval": VLLM_EPLB_STEP_INTERVAL,
                "num_redundant_experts": VLLM_EPLB_NUM_REDUNDANT_EXPERTS,
                "log_balancedness": VLLM_EPLB_LOG_BALANCEDNESS,
            }
        if data_parallel_size < 1:
            raise RuntimeError("data_parallel_size must be >= 1.")
        if enable_eplb and tp_size == 1 and data_parallel_size == 1:
            raise RuntimeError(
                "EPLB requires vLLM TP>1 or DP>1. "
                f"Got tp_size={tp_size}, data_parallel_size={data_parallel_size}."
            )
        world_size = tp_size * pp_size * data_parallel_size
        if devices is not None and len(devices) != world_size:
            raise RuntimeError(
                f"vLLM instance {idx} expects {world_size} devices "
                f"but got {len(devices)}."
            )
        instances.append(
            {
                "tp_size": tp_size,
                "pp_size": pp_size,
                "devices": devices,
                "engine_rank": engine_rank,
                "data_parallel_size": data_parallel_size,
                "enable_expert_parallel": enable_expert_parallel,
                "enable_eplb": enable_eplb,
                "expert_placement_strategy": expert_placement_strategy,
                "awex_cluster_id": awex_cluster_id,
                "eplb_config": eplb_config,
                "world_size": world_size,
            }
        )
    engine_ranks = [inst["engine_rank"] for inst in instances]
    if len(set(engine_ranks)) != len(engine_ranks):
        raise RuntimeError(f"Duplicate engine_rank values found: {engine_ranks}")
    instances.sort(key=lambda x: x["engine_rank"])
    expected = list(range(len(instances)))
    if engine_ranks != expected and [i["engine_rank"] for i in instances] != expected:
        raise RuntimeError(
            "engine_rank must be contiguous starting from 0 (e.g. 0,1,2...)."
        )
    return instances


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


def _http_get_json(url: str, timeout_s: int) -> dict:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def _http_post_json(url: str, payload: dict, timeout_s: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _discover_model_id(server_addr: str, fallback_model: str, timeout_s: int) -> str:
    try:
        payload = _http_get_json(f"http://{server_addr}/v1/models", timeout_s)
        models = payload.get("data") or []
        if models and isinstance(models[0], dict):
            model_id = models[0].get("id")
            if model_id:
                return str(model_id)
    except Exception as exc:
        print(
            f"[awex-test] failed to discover model id on {server_addr}: {exc}. "
            f"fallback to {fallback_model}",
            flush=True,
        )
    return fallback_model


def _drive_completion_traffic(
    *,
    server_addrs: list[str],
    fallback_model: str,
    requests_per_update: int,
    timeout_s: int,
    retries: int,
    strict: bool,
) -> None:
    if requests_per_update <= 0:
        return
    model_ids = {
        addr: _discover_model_id(
            addr, fallback_model=fallback_model, timeout_s=timeout_s
        )
        for addr in server_addrs
    }
    for server_addr in server_addrs:
        model_id = model_ids[server_addr]
        for req_idx in range(requests_per_update):
            payload = {
                "model": model_id,
                "prompt": f"awex eplb trigger request {req_idx}",
                "max_tokens": 8,
                "temperature": 0.0,
            }
            last_error = None
            for attempt in range(1, max(1, retries) + 1):
                try:
                    _http_post_json(
                        f"http://{server_addr}/v1/completions",
                        payload=payload,
                        timeout_s=timeout_s,
                    )
                    last_error = None
                    break
                except (
                    TimeoutError,
                    urllib.error.HTTPError,
                    urllib.error.URLError,
                    json.JSONDecodeError,
                    RuntimeError,
                ) as exc:
                    last_error = exc
                    time.sleep(min(1.0 * attempt, 3.0))
            if last_error is not None:
                message = (
                    f"[awex-test] completion traffic failed on {server_addr} after "
                    f"{max(1, retries)} attempts: {last_error}"
                )
                if strict:
                    raise RuntimeError(message) from last_error
                print(message, flush=True)
                return


def _run_awex_integration(result_queue, model_path: str):
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
            print(f"[awex-test] {name} timed out after {timeout}s.", flush=True)

    rank = 0
    world_size = 1
    try:
        device_backend = _detect_device_backend()
        os.environ.setdefault("AWEX_DEVICE_TYPE", device_backend)
        if device_backend == "npu":
            os.environ.setdefault("AWEX_USE_MINDSPEED", "1")
        os.environ.setdefault("AWEX_MASTER_ADDR", "127.0.0.1")
        train_world_size = _train_world_size()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if train_world_size > 1 and world_size == 1:
            raise RuntimeError(
                f"TRAIN_* sizes require WORLD_SIZE={train_world_size}. "
                "Launch with torchrun to enable multi-rank training."
            )
        if world_size > 1 and world_size != train_world_size:
            raise RuntimeError(
                f"WORLD_SIZE ({world_size}) does not match TRAIN_* world size ({train_world_size})."
            )

        vllm_instances = _build_vllm_instances()
        vllm_world_size = sum(inst["world_size"] for inst in vllm_instances)
        explicit_devices = [
            inst["devices"] for inst in vllm_instances if inst["devices"] is not None
        ]
        if explicit_devices and len(explicit_devices) != len(vllm_instances):
            raise RuntimeError(
                "VLLM_INSTANCES must either specify devices for all instances or none."
            )
        if explicit_devices:
            flat_vllm_devices = [d for inst in vllm_instances for d in inst["devices"]]
            if len(flat_vllm_devices) != vllm_world_size:
                raise RuntimeError(
                    f"Explicit vLLM devices ({len(flat_vllm_devices)}) do not match "
                    f"expected world size ({vllm_world_size})."
                )
            train_devices, _ = _select_devices(
                train_world_size=train_world_size,
                vllm_world_size=len(flat_vllm_devices),
                device_backend=device_backend,
                vllm_device_ids=flat_vllm_devices,
            )
        else:
            train_devices, flat_vllm_devices = _select_devices(
                train_world_size=train_world_size,
                vllm_world_size=vllm_world_size,
                device_backend=device_backend,
                vllm_device_ids=None,
            )
            offset = 0
            for inst in vllm_instances:
                size = inst["world_size"]
                inst["devices"] = flat_vllm_devices[offset : offset + size]
                offset += size

        train_device = train_devices[local_rank]

        # Start Awex meta server
        from awex.meta.meta_server import start_meta_server, stop_meta_server

        meta_server_addr = None

        # Launch vLLM server with Awex plugin enabled (real weights + validation).
        enable_validation = True
        comm_backend = os.environ.get("AREAL_AWEX_COMM_BACKEND", "nccl").lower()

        vllm_config = vLLMConfig(
            skip_tokenizer_init=False,
            model=model_path,
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            max_model_len=VLLM_MAX_MODEL_LEN,
            enforce_eager=True,
            # load_format="auto",
        )

        vllm_env_base = {
            "AWEX_DEVICE_TYPE": device_backend,
        }

        inf_engine = None
        train_engine = None
        tmpdir = None

        try:
            temp_config = InferenceEngineConfig(
                experiment_name="test_awex_megatron_vllm",
                trial_name="trial_0",
                setup_timeout=ENGINE_SETUP_TIMEOUT_S,
                request_timeout=ENGINE_REQUEST_TIMEOUT_S,
            )
            inf_engine = RemotevLLMEngine(temp_config)

            # Initialize Megatron engine
            if world_size == 1:
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

            # Pin Megatron to the first visible GPU
            if device_backend == "npu":
                if getattr(torch, "npu", None) is None:
                    raise RuntimeError("torch.npu is not available for NPU backend.")
                torch.npu.set_device(train_device)
            else:
                torch.cuda.set_device(train_device)

            train_config = TrainEngineConfig(
                experiment_name="test_awex_megatron_vllm",
                trial_name="trial_0",
                path=model_path,
                init_from_scratch=False,
                optimizer=None,
                megatron=MegatronEngineConfig(),
            )
            train_engine = MegatronEngine(train_config)
            train_parallel = ParallelStrategy(
                tensor_parallel_size=TRAIN_TP_SIZE,
                pipeline_parallel_size=TRAIN_PP_SIZE,
                data_parallel_size=TRAIN_DP_SIZE,
                context_parallel_size=TRAIN_CP_SIZE,
                expert_parallel_size=TRAIN_EP_SIZE,
                expert_tensor_parallel_size=TRAIN_ETP_SIZE,
            )
            train_engine.create_process_group(train_parallel)
            ft_spec = FinetuneSpec(
                total_train_epochs=1, dataset_size=128, train_batch_size=2
            )
            train_engine.initialize(addr=None, ft_spec=ft_spec)
            train_engine.set_version(1)
            if TRAIN_CLUSTER_ID:
                os.environ["AWEX_CLUSTER_ID"] = TRAIN_CLUSTER_ID

            if rank == 0:
                meta_ip, meta_port = start_meta_server()
                meta_server_addr = f"{meta_ip}:{meta_port}"
            if dist.is_initialized():
                obj_list = [meta_server_addr]
                dist.broadcast_object_list(obj_list, src=0)
                meta_server_addr = obj_list[0]

            server_addrs = None
            if rank == 0:
                server_infos = []
                for inst in vllm_instances:
                    vllm_args = vLLMConfig.build_args(
                        vllm_config=vllm_config,
                        tp_size=inst["tp_size"],
                        pp_size=inst["pp_size"],
                    )
                    vllm_extra_args = {
                        "data_parallel_size": inst.get("data_parallel_size"),
                        "enable_expert_parallel": inst.get("enable_expert_parallel"),
                    }
                    vllm_env = {
                        _visible_env_name(device_backend): ",".join(
                            map(str, inst["devices"])
                        ),
                        **vllm_env_base,
                    }
                    if inst.get("enable_eplb"):
                        if device_backend == "npu":
                            ascend_eplb_config = _build_ascend_eplb_config(
                                inst.get("eplb_config")
                            )
                            vllm_extra_args["additional_config"] = json.dumps(
                                {"eplb_config": ascend_eplb_config}
                            )
                            if ascend_eplb_config.get(
                                "dynamic_eplb"
                            ) or ascend_eplb_config.get("expert_map_record_path"):
                                vllm_env["DYNAMIC_EPLB"] = "true"
                            if ascend_eplb_config.get("expert_map_record_path"):
                                vllm_env["EXPERT_MAP_RECORD"] = "true"
                        else:
                            vllm_extra_args["enable_eplb"] = True
                            vllm_extra_args["eplb_config"] = json.dumps(
                                inst.get("eplb_config") or {}
                            )
                    placement = inst.get("expert_placement_strategy")
                    if placement and not (
                        device_backend == "npu" and inst.get("enable_eplb")
                    ):
                        vllm_extra_args["expert_placement_strategy"] = placement
                    vllm_args.update(vllm_extra_args)
                    cluster_id = inst.get("awex_cluster_id")
                    if cluster_id:
                        vllm_env["AWEX_CLUSTER_ID"] = str(cluster_id)
                    with _temp_env(vllm_env):
                        server_infos.append(inf_engine.launch_server(vllm_args))
                # RemoteInfEngine assigns engine_rank by the order of server_addrs.
                server_addrs = [f"{s.host}:{s.port}" for s in server_infos]
            if dist.is_initialized():
                obj_list = [server_addrs]
                dist.broadcast_object_list(obj_list, src=0)
                server_addrs = obj_list[0]
            # The inference engine's data-parallel size equals the number of vLLM
            # DP servers (one server_addr per DP rank). Pass it explicitly because
            # the test builds InferenceEngineConfig without a `backend` string, so
            # initialize() cannot derive it from self.config.backend.
            inf_engine.initialize(
                addr=server_addrs,
                train_data_parallel_size=len(server_addrs),
            )
            inf_engine.set_version(1)

            if enable_validation:
                # This integration test should validate every configured update by default.
                # If the env explicitly requests more validation steps, honor the larger value.
                validation_steps = max(
                    max(1, NUM_UPDATES),
                    VALIDATION_STEPS if VALIDATION_STEPS >= 0 else 0,
                )
                validate_every_n_steps = 1
                print(
                    "[awex-test] validation config: "
                    f"num_updates={NUM_UPDATES}, "
                    f"validation_steps={validation_steps}, "
                    f"validate_every_n_steps={validate_every_n_steps}",
                    flush=True,
                )
            else:
                validation_steps = 0
                validate_every_n_steps = max(1, VALIDATE_EVERY_N_STEPS)

            update_meta = WeightUpdateMeta.from_awex(
                meta_server_addr=meta_server_addr,
                comm_backend=(
                    "hccl"
                    if device_backend == "npu"
                    else ("file" if comm_backend == "file" else "nccl")
                ),
                # Force-disable MindSpeed for GPU runs; keep it enabled for NPU.
                use_mindspeed=(device_backend == "npu"),
                weights_validation_steps=validation_steps,
                validate_weights_every_n_steps=validate_every_n_steps,
                enable_debug_mode=enable_validation,
                debug_mode_config=(
                    {"raise_on_validation_fail": True} if enable_validation else {}
                ),
            )
            if device_backend == "npu":
                update_meta.weights_exchange_ipc_backend = "cpu"
            if comm_backend == "file":
                if rank == 0:
                    tmpdir = tempfile.TemporaryDirectory()
                    update_meta.path = tmpdir.name
                if dist.is_initialized():
                    obj_list = [update_meta.path]
                    dist.broadcast_object_list(obj_list, src=0)
                    update_meta.path = obj_list[0]

            train_engine.connect_engine(inf_engine, update_meta)
            for step in range(1, max(1, NUM_UPDATES) + 1):
                train_engine.set_version(step)
                inf_engine.set_version(step)
                if dist.is_initialized():
                    dist.barrier()
                train_engine.update_weights(update_meta)
                if dist.is_initialized():
                    dist.barrier()
                if rank == 0 and REQUESTS_PER_UPDATE > 0 and step < max(1, NUM_UPDATES):
                    _drive_completion_traffic(
                        server_addrs=server_addrs,
                        fallback_model=model_path,
                        requests_per_update=REQUESTS_PER_UPDATE,
                        timeout_s=max(1, REQUEST_TIMEOUT_S),
                        retries=max(1, REQUEST_RETRIES),
                        strict=STRICT_REQUEST_TRAFFIC,
                    )

            if result_queue is not None and rank == 0:
                result_queue.put(("ok", None))
        finally:
            if train_engine is not None:
                _safe_destroy(train_engine.destroy, "train_engine.destroy")
            if inf_engine is not None:
                _safe_destroy(inf_engine.destroy, "inf_engine.destroy")
            if tmpdir is not None:
                tmpdir.cleanup()
            if rank == 0 and meta_server_addr is not None:
                stop_meta_server()
            # Ensure any leftover subprocesses are cleaned up before returning.
            if rank == 0 and world_size == 1:
                kill_process_tree(os.getpid(), include_parent=False, graceful=False)
            if dist.is_initialized():
                dist.barrier()
    except Exception as exc:
        if result_queue is not None and rank == 0:
            result_queue.put(("error", repr(exc)))
        raise


def _resolve_moe_model_path() -> str:
    env_path = os.environ.get("AREAL_AWEX_MOE_MODEL_PATH")
    if env_path:
        if not os.path.exists(env_path):
            pytest.skip(
                f"AREAL_AWEX_MOE_MODEL_PATH not found: {env_path}. "
                "Set it to a reduced MoE checkpoint path."
            )
        return env_path
    if os.path.exists(MOE_LOCAL_PATH):
        return MOE_LOCAL_PATH
    pytest.skip(
        "Reduced MoE model not found locally. Build it with "
        "`tests/experimental/awex/build_reduced_qwen3_moe.py` and set "
        "AREAL_AWEX_MOE_MODEL_PATH."
    )


def _resolve_dense_model_path() -> str:
    env_path = os.environ.get("AREAL_AWEX_DENSE_MODEL_PATH")
    if env_path:
        if not os.path.exists(env_path):
            pytest.skip(
                f"AREAL_AWEX_DENSE_MODEL_PATH not found: {env_path}. "
                "Set it to a valid local path or enable download."
            )
        return env_path
    if os.path.exists(DENSE_LOCAL_PATH):
        return DENSE_LOCAL_PATH
    if os.environ.get("AREAL_AWEX_ALLOW_HF_DOWNLOAD") == "1":
        return get_model_path(DENSE_LOCAL_PATH, DENSE_HF_ID)
    pytest.skip(
        "Dense model not found locally. Set AREAL_AWEX_DENSE_MODEL_PATH or "
        "AREAL_AWEX_ALLOW_HF_DOWNLOAD=1 to download."
    )


@pytest.mark.slow
@pytest.mark.multi_gpu
def test_awex_megatron_to_vllm_nccl(tmp_path_factory):
    if not IS_VLLM_INSTALLED:
        pytest.skip("vLLM is not installed")
    if not IS_AWEX_INSTALLED:
        pytest.skip("awex is not installed")

    timeout_seconds = int(os.environ.get("AREAL_AWEX_TEST_TIMEOUT_S", "600"))
    model_kind = os.environ.get("AREAL_AWEX_MODEL", "dense").lower()
    if model_kind == "moe":
        model_path = _resolve_moe_model_path()
    elif model_kind == "dense":
        model_path = _resolve_dense_model_path()
    else:
        raise ValueError(
            f"Unknown AREAL_AWEX_MODEL value: {model_kind}. Use 'dense' or 'moe'."
        )
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        _run_awex_integration(None, model_path)
        return

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_run_awex_integration,
        args=(result_queue, model_path),
    )
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        kill_process_tree(proc.pid, graceful=False)
        proc.join(5)
        pytest.fail(f"Awex integration test exceeded {timeout_seconds}s timeout.")

    status = None
    detail = None
    try:
        status, detail = result_queue.get_nowait()
    except queue.Empty:
        status, detail = ("error", f"Worker exited with code {proc.exitcode}.")

    if status != "ok":
        pytest.fail(f"Awex integration test failed: {detail}")
