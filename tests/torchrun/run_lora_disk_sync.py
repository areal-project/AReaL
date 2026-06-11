"""Torchrun E2E validation for FSDP LoRA adapter-only disk saves."""

import argparse
import json
import os
import tempfile

import torch
import torch.distributed as dist
from safetensors.torch import load_file as safetensors_load_file

from areal.api import FinetuneSpec
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    FSDPEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import FSDPEngine
from areal.infra.platforms import current_platform

DEFAULT_MODEL_PATH = os.environ.get(
    "AREAL_LORA_DISK_SYNC_MODEL_PATH", "/workspace/models/Qwen3-0.6B"
)


_LORA_KEYWORDS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")


def write_result(path: str, success: bool) -> None:
    with open(path, "w") as f:
        f.write("Passed" if success else "Failed")


def make_fsdp_engine_with_lora(backend: str, model_path: str) -> FSDPEngine:
    """Create an FSDPEngine with LoRA + disk-mode weight update."""
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="test_lora_disk_sync",
        trial_name="test",
        mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
        path=model_path,
        optimizer=OptimizerConfig(),
        fsdp=FSDPEngineConfig(memory_efficient_load=True),
        # LoRA config
        use_lora=True,
        lora_rank=8,
        lora_alpha=16,
        peft_type="lora",
        # Disk-based weight update mode (LoRA disk sync is implicit
        # whenever use_lora=True and weight_update_mode="disk").
        weight_update_mode="disk",
    )
    alloc_mode = ModelAllocation.from_str(backend)
    engine = FSDPEngine(config)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=128, train_batch_size=8)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(None, ft_spec)
    return engine


def count_lora_and_base_params(engine: FSDPEngine):
    """Count LoRA vs base parameters in the engine model."""
    lora_count = 0
    base_count = 0
    for name, _param in engine.model.named_parameters():
        if any(kw in name for kw in _LORA_KEYWORDS):
            lora_count += 1
        else:
            base_count += 1
    return lora_count, base_count


def verify_forward_pass(engine: FSDPEngine) -> bool:
    """Run a simple forward pass to verify the model is still functional."""
    try:
        with torch.no_grad():
            engine.eval()
            input_ids = torch.randint(
                100, 1000, (2, 32), dtype=torch.long, device=engine.device
            )
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            outputs = engine.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            return logits is not None and logits.shape[0] == 2
    except Exception as e:
        print(f"Forward pass failed: {e}", flush=True)
        return False


def verify_adapter_artifacts(
    adapter_dir: str, *, lora_rank: int, lora_alpha: int
) -> bool:
    """Validate the PEFT-format files written by ``_save_model_to_hf``."""
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    config_path = os.path.join(adapter_dir, "adapter_config.json")

    if not os.path.exists(safetensors_path):
        print(f"ERROR: missing {safetensors_path}", flush=True)
        return False
    if os.path.getsize(safetensors_path) == 0:
        print(f"ERROR: empty {safetensors_path}", flush=True)
        return False
    if not os.path.exists(config_path):
        print(f"ERROR: missing {config_path}", flush=True)
        return False

    # Validate adapter_config.json content.
    with open(config_path) as f:
        cfg = json.load(f)
    if cfg.get("peft_type") != "LORA":
        print(f"ERROR: peft_type != LORA: {cfg.get('peft_type')}", flush=True)
        return False
    if cfg.get("task_type") != "CAUSAL_LM":
        print(f"ERROR: task_type != CAUSAL_LM: {cfg.get('task_type')}", flush=True)
        return False
    if cfg.get("r") != lora_rank:
        print(f"ERROR: r mismatch: {cfg.get('r')} vs {lora_rank}", flush=True)
        return False
    if cfg.get("lora_alpha") != lora_alpha:
        print(
            f"ERROR: lora_alpha mismatch: {cfg.get('lora_alpha')} vs {lora_alpha}",
            flush=True,
        )
        return False
    if "target_modules" not in cfg:
        print("ERROR: target_modules missing from adapter_config.json", flush=True)
        return False
    # ``base_model_name_or_path`` is required by SGLang's
    # /load_lora_adapter when it has to materialize the adapter on the
    # base model side.
    if "base_model_name_or_path" not in cfg:
        print(
            "ERROR: base_model_name_or_path missing from adapter_config.json",
            flush=True,
        )
        return False

    # Validate adapter_model.safetensors keys: every key must be a LoRA
    # tensor with the ``.default.`` segment stripped.
    state = safetensors_load_file(safetensors_path)
    if not state:
        print("ERROR: adapter_model.safetensors is empty", flush=True)
        return False
    for k in state:
        if not any(kw in k for kw in _LORA_KEYWORDS):
            print(f"ERROR: non-LoRA key found in adapter file: {k}", flush=True)
            return False
        if ".default." in k:
            print(f"ERROR: '.default.' was not stripped from key: {k}", flush=True)
            return False
        if not k.endswith(".weight"):
            print(f"ERROR: adapter key must end with '.weight': {k}", flush=True)
            return False

    # Catch accidental full-model saves.
    total_bytes = 0
    for root, _dirs, files in os.walk(adapter_dir):
        for fname in files:
            try:
                total_bytes += os.path.getsize(os.path.join(root, fname))
            except OSError:
                continue
    if total_bytes <= 0:
        print(f"ERROR: adapter directory total size is zero: {adapter_dir}", flush=True)
        return False
    if total_bytes > 200 * 1024 * 1024:
        print(
            f"ERROR: adapter directory size {total_bytes} bytes exceeds 200 MB ceiling -- "
            f"this likely means the engine fell back to a full-model save instead of "
            f"saving only the LoRA adapter.",
            flush=True,
        )
        return False
    print(
        f"[verify] adapter_dir={adapter_dir} total_bytes={total_bytes} "
        f"(adapter-only, well under full-model size)",
        flush=True,
    )
    return True


def test_lora_disk_sync(
    backend: str, output: str | None = None, model_path: str = DEFAULT_MODEL_PATH
) -> None:
    """Main test logic for LoRA disk sync."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    success = True

    alloc_mode = ModelAllocation.from_str(backend)
    dp = alloc_mode.parallel.dp_size
    tp = alloc_mode.parallel.tp_size

    print(
        f"[Rank {rank}] Starting LoRA disk sync test | "
        f"backend={backend} | dp={dp} tp={tp} | world_size={world_size}",
        flush=True,
    )

    # Step 1: Create engine with LoRA + disk mode.
    print(
        f"[Rank {rank}] Creating FSDP engine | dp={dp} tp={tp} "
        f"use_lora=True weight_update_mode=disk",
        flush=True,
    )
    engine = make_fsdp_engine_with_lora(backend, model_path)

    # Step 2: Verify LoRA parameters exist on the in-memory model.
    lora_count, base_count = count_lora_and_base_params(engine)
    print(
        f"[Rank {rank}] Model has {lora_count} LoRA params and "
        f"{base_count} base params",
        flush=True,
    )
    if lora_count == 0:
        print(f"[Rank {rank}] ERROR: No LoRA parameters found!", flush=True)
        success = False

    # Step 3: Verify config flags are correct.
    if not engine.config.use_lora:
        print(f"[Rank {rank}] ERROR: use_lora not set!", flush=True)
        success = False
    if engine.config.weight_update_mode != "disk":
        print(
            f"[Rank {rank}] ERROR: weight_update_mode != 'disk' "
            f"(got {engine.config.weight_update_mode})",
            flush=True,
        )
        success = False

    # Step 4: Trigger the actual save path used by
    # ``_update_weights_from_disk`` (no inference server required).
    with tempfile.TemporaryDirectory() as tmpdir:
        adapter_dir = os.path.join(tmpdir, "weight_update_v0")
        os.makedirs(adapter_dir, exist_ok=True)
        print(f"[Rank {rank}] Calling _save_model_to_hf -> {adapter_dir}", flush=True)
        engine._save_model_to_hf(
            adapter_dir,
            tokenizer=None,
            processor=None,
        )

        # Only rank 0 writes the artefacts; verify there.
        if rank == 0:
            ok = verify_adapter_artifacts(
                adapter_dir,
                lora_rank=engine.config.lora_rank,
                lora_alpha=engine.config.lora_alpha,
            )
            if not ok:
                success = False
            else:
                print(f"[Rank {rank}] PEFT adapter artefacts validated", flush=True)

    # Step 5: Verify forward pass still works after the save.
    print(f"[Rank {rank}] Verifying forward pass", flush=True)
    if not verify_forward_pass(engine):
        print(f"[Rank {rank}] ERROR: Forward pass failed!", flush=True)
        success = False
    else:
        print(f"[Rank {rank}] Forward pass OK", flush=True)

    # Cleanup.
    current_platform.synchronize()
    dist.barrier()
    engine.destroy()

    if rank == 0 and output:
        write_result(output, success)

    status = "PASSED" if success else "FAILED"
    print(f"[Rank {rank}] LoRA disk sync test {status}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Torchrun script for LoRA disk sync e2e test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save test result (Passed/Failed)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fsdp:d1t1",
        help="Backend allocation string (e.g., 'fsdp:d1t1')",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Local model path used to initialize the FSDP engine.",
    )
    args = parser.parse_args()
    test_lora_disk_sync(
        backend=args.backend, output=args.output, model_path=args.model_path
    )


if __name__ == "__main__":
    main()
