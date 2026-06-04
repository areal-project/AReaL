"""Distributed worker for FSDP optimizer dtype invariants test.

Asserts:
  1. Underlying param storage matches optimizer_dtype.
  2. AdamW optimizer states (exp_avg, exp_avg_sq) match optimizer_dtype.
  3. Forward output matches compute dtype (config.dtype).
  4. After save_pretrained, exported safetensors weights are in compute dtype.
  5. `_cast_to_compute_dtype(_get_full_tensor(...))` yields compute dtype (xccl path).

Launched via torchrun by tests/test_fsdp_optimizer_dtype.py.
"""

import argparse
import os
import shutil
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from areal.api import FinetuneSpec
from areal.api.cli_args import (
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import FSDPEngine
from areal.infra.platforms import current_platform


def _resolve_model_path(local_path: str, hf_id: str) -> str:
    """Inline get_model_path, bypassing tests.utils.

    Importing tests.utils → areal.utils.testing_utils triggers
    module-level snapshot_download for 8 models (incl. 30B+ MoE),
    which can OOM or fill disk in environments without those models
    pre-cached. Use this lightweight version to fetch only the model
    this worker actually needs.
    """
    if os.path.exists(local_path):
        return local_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=hf_id, ignore_patterns=["*.gguf", "*.ggml"])


def setup_distributed():
    if dist.is_initialized():
        return
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )
    current_platform.set_device(rank)


def get_local_dtype(t: torch.Tensor) -> torch.dtype:
    """Extract dtype from DTensor or regular tensor."""
    if isinstance(t, DTensor):
        return t.to_local().dtype
    return t.dtype


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--optimizer_dtype", default="float32")
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model_path = _resolve_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
    )

    config = TrainEngineConfig(
        experiment_name="test",
        trial_name="test",
        path=model_path,
        backend=f"fsdp:d{world_size}p1t1",
        dtype=args.dtype,
        optimizer_dtype=args.optimizer_dtype,
        gradient_checkpointing=False,
        # sdpa avoids depending on flash_attn (which may be missing in some
        # environments); dtype invariants are independent of attention impl.
        attn_impl="sdpa",
        mb_spec=MicroBatchSpec(max_tokens_per_mb=512),
        optimizer=OptimizerConfig(
            type="adam",
            lr=1e-4,
            lr_scheduler_type="constant",
        ),
    )

    # Resolve parallel strategy from backend string ("fsdp:d{N}p1t1").
    # initialize() ignores parallel_strategy kwargs; the world mesh and
    # parallel_helper must be set up via create_process_group() first.
    from areal.api.alloc_mode import ModelAllocation

    alloc = ModelAllocation.from_str(config.backend)

    engine = FSDPEngine(config)
    engine.create_process_group(parallel_strategy=alloc.parallel)
    engine.initialize(
        addr=None,
        ft_spec=FinetuneSpec(total_train_epochs=1, dataset_size=10, train_batch_size=1),
    )

    expected_storage = getattr(torch, args.optimizer_dtype)
    expected_compute = getattr(torch, args.dtype)

    # ---- Invariant 1: underlying param storage = optimizer_dtype ----
    sample_param = next(p for p in engine.model.parameters() if p.requires_grad)
    storage_dtype = get_local_dtype(sample_param)
    assert storage_dtype == expected_storage, (
        f"[rank {rank}] param storage dtype {storage_dtype} != "
        f"expected {expected_storage}"
    )

    # ---- Run a step to populate optimizer state ----
    seq_len = 64
    input_ids = torch.randint(
        0, 1000, (1, seq_len), device=engine.device, dtype=torch.long
    )
    attention_mask = torch.ones((1, seq_len), device=engine.device, dtype=torch.long)
    engine.optimizer_zero_grad()
    out = engine.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=torch.arange(seq_len, device=engine.device).unsqueeze(0),
    )
    logits = out.logits if hasattr(out, "logits") else out

    # ---- Invariant 3: compute output matches compute dtype ----
    assert logits.dtype == expected_compute, (
        f"[rank {rank}] forward output dtype {logits.dtype} != "
        f"expected {expected_compute}"
    )

    loss = logits.float().mean()
    loss.backward()
    engine.optimizer_step()

    # ---- Invariant 2: optimizer states match optimizer_dtype ----
    state = engine.optimizer.state[sample_param]
    for key in ["exp_avg", "exp_avg_sq"]:
        assert key in state, f"[rank {rank}] optimizer missing key {key}"
        sd = get_local_dtype(state[key])
        assert sd == expected_storage, (
            f"[rank {rank}] optimizer state[{key}].dtype {sd} != "
            f"expected {expected_storage}"
        )

    # ---- Invariant 4: HF export uses compute dtype ----
    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="areal_dtype_test_")
    else:
        tmpdir = None
    obj_list = [tmpdir]
    dist.broadcast_object_list(obj_list, src=0)
    tmpdir = obj_list[0]

    try:
        engine._save_model_to_hf(tmpdir, engine.tokenizer, engine.processor)

        if rank == 0:
            from safetensors import safe_open

            # Find any safetensors file
            files = [f for f in os.listdir(tmpdir) if f.endswith(".safetensors")]
            assert files, f"No safetensors files in {tmpdir}: {os.listdir(tmpdir)}"
            with safe_open(os.path.join(tmpdir, files[0]), framework="pt") as sf:
                for k in sf.keys():
                    t = sf.get_tensor(k)
                    if t.is_floating_point():
                        assert t.dtype == expected_compute, (
                            f"HF export: {k}.dtype {t.dtype} != {expected_compute}"
                        )
                        break  # one float param is sufficient
    finally:
        # Clean up rank-0 tmpdir (mkdtemp is not auto-cleaned).
        # _save_model_to_hf already issues dist.barrier(group=cpu_group)
        # before returning, so file handles are released by the time we rmtree.
        if rank == 0 and tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- Invariant 5: shared cast helper (xccl weight sync path) ----
    # No SGLang round-trip in this test; call the same helper Task 4b uses.
    full_tensor = engine._get_full_tensor(sample_param)
    cast_tensor = engine._cast_to_compute_dtype(full_tensor)
    assert cast_tensor.dtype == expected_compute, (
        f"[rank {rank}] _cast_to_compute_dtype: {cast_tensor.dtype} "
        f"!= compute {expected_compute}"
    )

    dist.barrier()
    if rank == 0:
        print(
            f"[OK] dtype={args.dtype} optimizer_dtype={args.optimizer_dtype} "
            f"world_size={world_size}"
        )

    engine.destroy()


if __name__ == "__main__":
    main()
