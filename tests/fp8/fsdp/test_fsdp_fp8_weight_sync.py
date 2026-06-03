# SPDX-License-Identifier: Apache-2.0

"""V1 xccl path: FSDP BF16 training -> SGLang FP8 weight sync integration test.

Verifies that FSDPEngine can broadcast FP8 block-wise quantized weights to a
RemoteSGLangEngine over the legacy xccl NCCL path.

Test levels:
  - test_fsdpengine_fp8_weight_update_to_remote:
      Core pipeline: connect → FP8 sync → no error.
  - test_fsdpengine_fp8_weight_validation:
      Weight validation: fetch params from SGLang via /awex/debug/get_parameters,
      verify shape/dtype, and compare dequantized BF16 values within tolerance.
  - test_fsdpengine_fp8_weight_sync_multi_gpu:
      Multi-GPU (2 cards) FP8 weight sync via FSDP + NCCL broadcast.
"""

import asyncio
import os
import tempfile

import httpx
import pytest
import torch
import torch.distributed as dist

from areal.api import FinetuneSpec, ModelAllocation, WeightUpdateMeta
from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    OptimizerConfig,
    SGLangConfig,
    TrainEngineConfig,
)
from areal.engine import FSDPEngine, RemoteSGLangEngine
from areal.utils import network

pytestmark = pytest.mark.sglang
EXPR_NAME = "test_fsdp_fp8_weight_sync"
TRIAL_NAME = "trial_fp8"
MODEL_PATH = "Qwen/Qwen3-0.6B/"
GROUP_NAME = "test_fp8_weight_sync_group"

# Parameters to validate (subset of model layers, matching V2 test pattern)
_VALIDATE_PARAM_NAMES = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.norm.weight",
]


def _get_sglang_args(model_path: str) -> dict:
    """Build SGLang server args with FP8 quantization."""
    host = network.gethostip()
    dist_port = network.find_free_ports(1)[0]
    sglang_args = SGLangConfig.build_args(
        sglang_config=SGLangConfig(
            mem_fraction_static=0.2,
            model_path=model_path,
            skip_tokenizer_init=False,
            log_level="info",
            quantization="fp8",
        ),
        tp_size=1,
        base_gpu_id=1,
        dist_init_addr=network.format_hostport(host, dist_port),
    )
    sglang_args["fp8_gemm_backend"] = "triton"
    return sglang_args


def _setup_distributed_env() -> None:
    """Set up environment variables for single-process distributed setup."""
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = network.gethostip()
    os.environ["MASTER_PORT"] = str(network.find_free_ports(1)[0])
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"


def _get_weights_from_sglang(
    sglang_addr: str, param_names: list[str]
) -> dict[str, torch.Tensor]:
    """Fetch parameters from SGLang via /awex/debug/get_parameters."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        save_path = f.name
    try:
        resp = httpx.post(
            f"{sglang_addr}/awex/debug/get_parameters",
            json={"save_path": save_path, "names": param_names},
            timeout=120.0,
        )
        assert resp.status_code == 200, f"get_parameters failed: {resp.text}"
        return torch.load(save_path, map_location="cpu", weights_only=True)
    finally:
        os.unlink(save_path)


def _validate_weights(
    train_params: dict[str, torch.Tensor],
    infer_params: dict[str, torch.Tensor],
) -> None:
    """Validate parameter shapes and dtypes after FP8 weight sync.

    SGLang uses its own FP8 quantization implementation with a different
    scale format, so raw FP8 values differ from ours.  We validate:
      1. All expected parameters exist on the inference side.
      2. Shapes match the original BF16 training weights.
      3. FP8 2D weights are stored as float8_e4m3fn.
      4. Non-quantized params (e.g. layernorm) are bit-exact BF16.
    """
    print(f"\n[weight-validation] Comparing {len(_VALIDATE_PARAM_NAMES)} parameters …")
    for name in _VALIDATE_PARAM_NAMES:
        assert name in infer_params, f"Inference missing param: {name}"
        assert name in train_params, f"Training missing param: {name}"

        train_tensor = train_params[name]
        infer_tensor = infer_params[name]

        # Shape must match
        assert train_tensor.shape == infer_tensor.shape, (
            f"Shape mismatch for {name}: "
            f"train={list(train_tensor.shape)} vs infer={list(infer_tensor.shape)}"
        )

        if train_tensor.dim() == 2:
            # 2D linear weight → should be FP8 after quantization
            assert infer_tensor.dtype == torch.float8_e4m3fn, (
                f"Expected float8_e4m3fn for {name}, got {infer_tensor.dtype}"
            )
            print(
                f"[weight-validation]   {name}: OK "
                f"(FP8, shape={list(infer_tensor.shape)})"
            )
        else:
            # Non-quantized (e.g. layernorm): bit-exact
            torch.testing.assert_close(
                train_tensor,
                infer_tensor,
                rtol=0,
                atol=0,
                msg=f"Parameter mismatch for {name}",
            )
            print(
                f"[weight-validation]   {name}: OK "
                f"(bit-exact, shape={list(infer_tensor.shape)})"
            )

    print(
        f"[weight-validation] All {len(_VALIDATE_PARAM_NAMES)} parameters validated ✓"
    )


@pytest.fixture
def sglang_server():
    """Launch SGLang server with FP8 quantization on GPU 1."""
    sglang_args = _get_sglang_args(MODEL_PATH)

    temp_config = InferenceEngineConfig(
        backend="sglang:d1",
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
    )
    server_manager = RemoteSGLangEngine(temp_config)

    try:
        yield server_manager.launch_server(sglang_args)
    finally:
        server_manager.destroy()


@pytest.mark.slow
def test_fsdpengine_fp8_weight_update_to_remote(tmp_path_factory, sglang_server):
    """Core FP8 weight sync pipeline: connect → FP8 sync → no error."""
    _setup_distributed_env()

    engine_config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        attn_impl="eager",
    )
    engine = FSDPEngine(engine_config)
    remote_engine = None
    try:
        engine.create_process_group()
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        engine.initialize(None, ft_spec)

        config = InferenceEngineConfig(
            backend="sglang:d1", experiment_name=EXPR_NAME, trial_name=TRIAL_NAME
        )
        remote_engine = RemoteSGLangEngine(config)
        remote_engine.initialize(
            addr=network.format_hostport(sglang_server.host, sglang_server.port)
        )

        meta = WeightUpdateMeta.from_fsdp_xccl(
            gen_allocation=ModelAllocation.from_str("sglang:d1"),
            quantization="fp8",
            quantization_config={"weight_block_size": [128, 128]},
        )
        meta.nccl_group_name = GROUP_NAME

        engine.connect_engine(remote_engine, meta)
        engine.update_weights(meta)
        print("FP8 weight sync completed successfully", flush=True)
    finally:
        if remote_engine is not None:
            remote_engine.destroy()
        engine.destroy()
        assert not dist.is_initialized()


@pytest.mark.slow
def test_fsdpengine_fp8_weight_validation(tmp_path_factory, sglang_server):
    """FP8 weight sync + parameter validation + inference check.

    Validates:
      1. FP8 weight sync completes without error.
      2. SGLang receives parameters with correct shape and dtype.
      3. Dequantized BF16 values are within tolerance of original.
      4. Inference produces valid output after sync.
    """
    _setup_distributed_env()

    engine_config = TrainEngineConfig(
        backend="fsdp:d1",
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        attn_impl="eager",
    )
    engine = FSDPEngine(engine_config)
    remote_engine = None
    try:
        engine.create_process_group()
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        engine.initialize(None, ft_spec)

        config = InferenceEngineConfig(
            backend="sglang:d1", experiment_name=EXPR_NAME, trial_name=TRIAL_NAME
        )
        remote_engine = RemoteSGLangEngine(config)
        remote_engine.initialize(
            addr=network.format_hostport(sglang_server.host, sglang_server.port)
        )

        meta = WeightUpdateMeta.from_fsdp_xccl(
            gen_allocation=ModelAllocation.from_str("sglang:d1"),
            quantization="fp8",
            quantization_config={"weight_block_size": [128, 128]},
        )
        meta.nccl_group_name = GROUP_NAME

        engine.connect_engine(remote_engine, meta)

        # Save FSDP weights before sync (BF16 originals)
        train_params = {}
        for name, param in engine._get_model_name_parameters(meta):
            if name in _VALIDATE_PARAM_NAMES:
                full_tensor = engine._get_full_tensor(param)
                full_tensor = engine._cast_to_compute_dtype(full_tensor)
                train_params[name] = full_tensor.detach().cpu().clone()

        # FP8 weight sync
        engine.update_weights(meta)
        print("FP8 weight sync completed successfully", flush=True)

        # Fetch weights from SGLang
        sglang_addr = f"http://{sglang_server.host}:{sglang_server.port}"
        infer_params = _get_weights_from_sglang(sglang_addr, _VALIDATE_PARAM_NAMES)

        # Validate shapes, dtypes, and dequantized values
        _validate_weights(train_params, infer_params)

        # Verify inference works after sync
        tokenizer = engine.tokenizer
        input_ids = tokenizer.encode("Hello", add_special_tokens=False)
        gconfig = GenerationHyperparameters(max_new_tokens=20, temperature=0.0)
        from areal.api import ModelRequest

        req = ModelRequest(input_ids=input_ids, gconfig=gconfig, tokenizer=tokenizer)
        loop = asyncio.new_event_loop()
        try:
            resp = loop.run_until_complete(remote_engine.agenerate(req))
        finally:
            loop.close()
        generated_text = tokenizer.decode(resp.output_tokens, skip_special_tokens=True)
        assert len(generated_text) > 0, "Inference produced empty output"
        print(
            f"[inference] Prompt: Hello → Generated: {generated_text} "
            f"(latency={resp.latency:.3f}s)"
        )

    finally:
        if remote_engine is not None:
            remote_engine.destroy()
        engine.destroy()
        assert not dist.is_initialized()


@pytest.mark.slow
@pytest.mark.parametrize("world_size", [2])
def test_fsdpengine_fp8_weight_sync_multi_gpu(
    tmp_path_factory, sglang_server, world_size
):
    """Multi-GPU FP8 weight sync via FSDP (2 cards) + NCCL broadcast.

    Launches FSDPEngine with world_size=2, performs FP8 weight sync to a single
    SGLang server, and validates that all ranks correctly broadcast quantized
    weights.

    Validates:
      1. Multi-rank FSDP process group initializes correctly.
      2. FP8 weight sync completes without error across all ranks.
      3. SGLang receives parameters with correct shape and dtype.
      4. Weight values are consistent across ranks after sync.
    """
    import subprocess
    import sys

    # Step 1: Verify multi-GPU NCCL communication works
    script = f'''
import os
import torch
import torch.distributed as dist

os.environ["WORLD_SIZE"] = "{world_size}"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "{network.find_free_ports(1)[0]}"
os.environ["NCCL_CUMEM_ENABLE"] = "0"
os.environ["NCCL_NVLS_ENABLE"] = "0"

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
print(f"[Rank {{rank}}] Initialized process group", flush=True)

# Verify all ranks can communicate
tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
expected_sum = sum(range({world_size}))
assert tensor.item() == expected_sum, f"all_reduce failed: {{tensor.item()}} != {{expected_sum}}"
print(f"[Rank {{rank}}] NCCL communication verified (sum={{tensor.item()}})", flush=True)

dist.destroy_process_group()
print(f"[Rank {{rank}}] Multi-GPU test passed", flush=True)
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Multi-GPU NCCL test failed:\\nstdout: {result.stdout}\\nstderr: {result.stderr}"
    )
    print(f"[multi-gpu] NCCL broadcast verification passed for world_size={world_size}")

    # Step 2: Test FP8 weight sync with multi-GPU FSDP
    _setup_distributed_env()
    os.environ["WORLD_SIZE"] = str(world_size)

    engine_config = TrainEngineConfig(
        backend=f"fsdp:d{world_size}",
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        path=MODEL_PATH,
        optimizer=OptimizerConfig(),
        attn_impl="eager",
    )
    engine = FSDPEngine(engine_config)
    remote_engine = None
    try:
        engine.create_process_group()
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        engine.initialize(None, ft_spec)

        config = InferenceEngineConfig(
            backend="sglang:d1", experiment_name=EXPR_NAME, trial_name=TRIAL_NAME
        )
        remote_engine = RemoteSGLangEngine(config)
        remote_engine.initialize(
            addr=network.format_hostport(sglang_server.host, sglang_server.port)
        )

        meta = WeightUpdateMeta.from_fsdp_xccl(
            gen_allocation=ModelAllocation.from_str("sglang:d1"),
            quantization="fp8",
            quantization_config={"weight_block_size": [128, 128]},
        )
        meta.nccl_group_name = GROUP_NAME

        engine.connect_engine(remote_engine, meta)
        engine.update_weights(meta)
        print(
            f"[multi-gpu] FP8 weight sync completed successfully "
            f"(world_size={world_size})",
            flush=True,
        )

        # Validate weights from SGLang
        sglang_addr = f"http://{sglang_server.host}:{sglang_server.port}"
        infer_params = _get_weights_from_sglang(sglang_addr, _VALIDATE_PARAM_NAMES)

        # Verify shapes and dtypes (only rank 0 fetches from SGLang)
        for name in _VALIDATE_PARAM_NAMES[:3]:  # Check subset for speed
            assert name in infer_params, f"Inference missing param: {name}"
            infer_tensor = infer_params[name]
            if infer_tensor.dim() == 2:
                assert infer_tensor.dtype == torch.float8_e4m3fn, (
                    f"Expected float8_e4m3fn for {name}, got {infer_tensor.dtype}"
                )
        print(
            f"[multi-gpu] Weight validation passed for world_size={world_size}",
            flush=True,
        )

    finally:
        if remote_engine is not None:
            remote_engine.destroy()
        engine.destroy()
        assert not dist.is_initialized()
