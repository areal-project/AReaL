# SPDX-License-Identifier: Apache-2.0

"""Distributed integration tests for the Megatron VLM path.

All tests in this file launch ``tests/torchrun/run_megatron_engine_vlm_distributed.py``
as a torchrun subprocess so the parent pytest process never allocates GPU
memory. The full suite runs on as few as 2 devices for the dense parametric
tests; Qwen3-VL-MoE tests need 8.

CPU-only unit tests for the converters / detection helpers live in
``tests/test_megatron_engine_vlm.py``.
"""

import os
import pathlib
import subprocess
import sys

import pytest

try:
    import mindspeed.megatron_adaptor  # noqa: F401 isort: skip  # must precede mbridge on NPU
except ImportError:
    pass

from areal.api.alloc_mode import ModelAllocation
from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports
from areal.utils.testing_utils import DENSE_MODEL_PATHS, MOE_MODEL_PATHS

_TORCHRUN_SCRIPT = (
    pathlib.Path(__file__).parent
    / "torchrun"
    / "run_megatron_engine_vlm_distributed.py"
).resolve()

# Detect any accelerator (CUDA or NPU)
ACCELERATOR_AVAILABLE = current_platform.device_type in ("cuda", "npu")
# DCP-format checkpointing is unsupported on NPU: the vendored
# megatron-core v0.12.1 calls torch's private ``_write_item`` with the
# pre-``serialization_format`` signature, which fails on torch 2.9+
# shipped in NPU containers.
NPU_DCP_UNSUPPORTED = current_platform.device_type == "npu"


def _run_vlm_test(
    test_type: str,
    output_path: str,
    *,
    backend: str = "megatron:d1p1t1",
    nproc: int | None = None,
    extra_args: list[str] | None = None,
    timeout: int = 1800,
    env_overrides: dict[str, str] | None = None,
):
    """Launch a VLM integration test via torchrun subprocess.

    ``nproc`` is inferred from ``ModelAllocation.from_str(backend).parallel.world_size``
    when not provided, so asymmetric ``(attn:...|ffn:...)`` allocations work
    without callers having to compute the rank count separately. Pass
    ``env_overrides={"VLM_MODEL_PATH": ...}`` to switch the model under test.

    Output is streamed to the parent's stdout/stderr (helpful for diagnosing
    long distributed runs); OOM detection happens via the runner's own
    ``write_result(output, "OOM")`` marker rather than stderr scraping.
    """
    if nproc is None:
        nproc = ModelAllocation.from_str(backend).parallel.world_size

    port = find_free_ports(1)[0]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(nproc))
    if env_overrides:
        env.update(env_overrides)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        str(_TORCHRUN_SCRIPT),
        f"--backend={backend}",
        f"--test_type={test_type}",
        f"--output={output_path}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    output_path_obj = pathlib.Path(output_path)
    try:
        subprocess.run(
            cmd,
            env=env,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stdout,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        if output_path_obj.exists() and output_path_obj.read_text().strip() == "OOM":
            pytest.skip(f"OOM: VLM {test_type} requires more GPU memory")
        pytest.fail(f"VLM {test_type} test failed (exit {e.returncode})")
    except subprocess.TimeoutExpired:
        pytest.fail(f"VLM {test_type} test timed out ({timeout}s)")

    result = output_path_obj.read_text().strip()
    if result == "OOM":
        pytest.skip(f"OOM: VLM {test_type} requires more GPU memory")
    assert result == "Passed", f"VLM {test_type} test failed: {result}"


# ──────────────────────────────────────────────────────────────────────
# Per-model scenarios.
#
# Each entry is ``{"env": <env-vars>, "backend": <override>}``. ``backend``
# is ``None`` for dense VLMs (the test's own default backend is used) and
# set explicitly for MoE entries that can't fit the default allocation —
# Qwen3-VL-MoE-30B-A3B OOMs on ``d1p1t1`` (and on TP-only allocations: TP
# doesn't shard the expert dimension, so the 128 routed experts per layer
# stay fully replicated on every rank). The asymmetric ``(attn:d2t4|ffn:d2e4)``
# allocation puts experts on EP=4 which is the smallest allocation that
# both fits and exercises the EP code path.
#
# To add a new VLM, register the path in ``areal.utils.testing_utils`` and
# append one entry here — no test-body changes needed.
# ──────────────────────────────────────────────────────────────────────

_VLM_MODELS = [
    pytest.param(
        {
            "env": {"VLM_MODEL_PATH": DENSE_MODEL_PATHS["qwen2_5_vl"]},
            "backend": None,
        },
        id="qwen25_vl",
    ),
    pytest.param(
        {
            "env": {"VLM_MODEL_PATH": DENSE_MODEL_PATHS["qwen3_vl"]},
            "backend": None,
        },
        id="qwen3_vl",
    ),
    pytest.param(
        {
            "env": {"VLM_MODEL_PATH": MOE_MODEL_PATHS["qwen3_vl_moe"]},
            "backend": "megatron:(attn:d2t4|ffn:d2e4)",
        },
        id="qwen3_vl_moe",
        marks=pytest.mark.skipif(
            current_platform.device_count() < 8,
            reason="Qwen3-VL-MoE-30B-A3B requires at least 8 GPUs (EP=4)",
        ),
    ),
]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not ACCELERATOR_AVAILABLE, reason="No accelerator available")
@pytest.mark.parametrize("model_env", _VLM_MODELS)
def test_engine_initializes(model_env, tmp_path_factory):
    """Verify VLM engine detects vision model and loads processor."""
    output = str(tmp_path_factory.mktemp("vlm_test") / "init.out")
    _run_vlm_test(
        "init",
        output,
        backend=model_env["backend"] or "megatron:d1p1t1",
        env_overrides=model_env["env"],
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not ACCELERATOR_AVAILABLE, reason="No accelerator available")
@pytest.mark.parametrize("model_env", _VLM_MODELS)
def test_simple_forward(model_env, tmp_path_factory):
    """Verify forward pass with VLM inputs completes."""
    output = str(tmp_path_factory.mktemp("vlm_test") / "forward.out")
    _run_vlm_test(
        "forward",
        output,
        backend=model_env["backend"] or "megatron:d1p1t1",
        env_overrides=model_env["env"],
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not ACCELERATOR_AVAILABLE, reason="No accelerator available")
@pytest.mark.parametrize("model_env", _VLM_MODELS)
def test_hf_save_load_weights(model_env, tmp_path_factory):
    """Verify save/load preserves VLM weights and saves processor."""
    save_dir = str(tmp_path_factory.mktemp("vlm_save"))
    output = str(tmp_path_factory.mktemp("vlm_test") / "save_load.out")
    _run_vlm_test(
        "save_load",
        output,
        backend=model_env["backend"] or "megatron:d1p1t1",
        extra_args=[f"--save_dir={save_dir}"],
        env_overrides=model_env["env"],
    )


@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not ACCELERATOR_AVAILABLE, reason="No accelerator available")
@pytest.mark.parametrize("model_env", _VLM_MODELS)
def test_train_tensor_parallel(model_env, tmp_path_factory):
    """VLM training with TP=2 (dense) / EP-aware allocation (MoE) to avoid OOM."""
    if current_platform.device_count() < 2:
        pytest.skip("VLM TP training requires at least 2 GPUs")
    output = str(tmp_path_factory.mktemp("vlm_test") / "train_tp2.out")
    _run_vlm_test(
        "train",
        output,
        backend=model_env["backend"] or "megatron:d1p1t2",
        env_overrides=model_env["env"],
    )


# ──────────────────────────────────────────────────────────────────────
# Qwen3-VL-MoE: 30B-A3B-Instruct under hybrid (attn|ffn) allocation.
# CP > 1 is forbidden for VLMs (megatron_engine.py:347).
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not ACCELERATOR_AVAILABLE, reason="No accelerator available")
def test_qwen3vl_moe_expert_parallel(tmp_path_factory):
    """Forward smoke test for Qwen3-VL-MoE under ``(attn:d2t4|ffn:d2e4)``.

    Allocation: attn DP=2 TP=4 (8 GPUs); ffn DP=2 EP=4 (8 GPUs). World sizes
    must match — earlier ``(attn:d2t2|ffn:d2e4)`` raised
    ``InvalidAllocationModeError`` (attn=4 vs ffn=8). Validates the hybrid
    attn/ffn parser, EP-aware weight init, and VLM forward path.
    """
    if current_platform.device_count() < 8:
        pytest.skip("Qwen3-VL-MoE expert parallel requires 8 GPUs to run")
    output = str(
        tmp_path_factory.mktemp("test_output") / "qwen3vl_moe_expert_parallel.out"
    )
    _run_vlm_test(
        "forward",
        output,
        backend="megatron:(attn:d2t4|ffn:d2e4)",
        env_overrides={"VLM_MODEL_PATH": MOE_MODEL_PATHS["qwen3_vl_moe"]},
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not ACCELERATOR_AVAILABLE, reason="No accelerator available")
def test_qwen3vl_moe_hf_save_load(tmp_path_factory):
    """HF save/load round-trip for Qwen3-VL-MoE under ``(attn:d2t4|ffn:d2e4)``.

    Mirrors the forward smoke test's allocation (8 devices) and adds an
    end-to-end ``save → zero → load → forward-match`` cycle in HF
    safetensors format. This exercises the MoE expert collection / gather
    path in ``hf_save.py`` for 30B-A3B that DCP save/load can't cover on
    NPU (vendored mcore 0.12.1 has a torch-2.9 signature mismatch).
    """
    if current_platform.device_count() < 8:
        pytest.skip("Qwen3-VL-MoE HF save load requires 8 GPUs to run")
    save_dir = str(tmp_path_factory.mktemp("vlm_save_moe"))
    output = str(
        tmp_path_factory.mktemp("test_output") / "qwen3vl_moe_hf_save_load.out"
    )
    _run_vlm_test(
        "save_load",
        output,
        backend="megatron:(attn:d2t4|ffn:d2e4)",
        extra_args=[f"--save_dir={save_dir}"],
        env_overrides={"VLM_MODEL_PATH": MOE_MODEL_PATHS["qwen3_vl_moe"]},
    )


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not ACCELERATOR_AVAILABLE, reason="No accelerator available")
@pytest.mark.skipif(
    NPU_DCP_UNSUPPORTED,
    reason="DCP save/load unsupported on NPU (mcore 0.12.1 + torch 2.9 _write_item "
    "signature mismatch); skip until vendored mcore is bumped or monkey-patched.",
)
def test_qwen3vl_moe_dcp_save_load(tmp_path_factory):
    """DCP save/load round-trip for Qwen3-VL-MoE under ``(attn:d2p1t4|ffn:d1p1t2e4)``.

    Allocation: attn DP=2 PP=1 TP=4 (8 GPUs); ffn DP=1 PP=1 TP=2 EP=4 (8 GPUs).
    Drops the ``cp=2`` segment from the dense Qwen3-MoE analog because VLM
    forbids CP>1 (megatron_engine.py:347).
    """
    if current_platform.device_count() < 8:
        pytest.skip("Qwen3-VL-MoE DCP save load requires 8 GPUs to run")
    output = str(tmp_path_factory.mktemp("test_output") / "qwen3vl_moe_save_load.out")
    _run_vlm_test(
        "dcp_save_load",
        output,
        backend="megatron:(attn:d2p1t4|ffn:d1p1t2e4)",
        env_overrides={"VLM_MODEL_PATH": MOE_MODEL_PATHS["qwen3_vl_moe"]},
    )
