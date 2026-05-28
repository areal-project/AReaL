"""Torchrun script for Megatron VLM integration tests.

Launched via torchrun from test_megatron_engine_vlm.py. All integration
tests run as subprocesses so the parent pytest process never allocates
GPU memory, allowing the full suite to run on just 2 GPUs.

Supports any registered VLM whose ``hf_config`` exposes ``vision_config``
with ``patch_size``, ``temporal_patch_size``, ``spatial_merge_size``,
``in_channels``, and ``image_token_id`` — ``mock_vlm_input`` reads patch
geometry from ``engine.hf_config`` so this script works for both
Qwen2.5-VL (patch=14) and Qwen3-VL (patch=16) without code-side
branching. Pick the model via the ``VLM_MODEL_PATH`` env var; default
is ``DENSE_MODEL_PATHS["qwen2_5_vl"]`` from ``areal.utils.testing_utils``
(local path preferred, HF Hub fallback).
"""

import argparse
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from areal.api import FinetuneSpec, SaveLoadMeta
from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    MegatronEngineConfig,
    MicroBatchSpec,
    OptimizerConfig,
    TrainEngineConfig,
)
from areal.engine import MegatronEngine
from areal.utils.data import broadcast_tensor_container
from areal.utils.testing_utils import DENSE_MODEL_PATHS

VLM_MODEL_PATH = os.environ.get("VLM_MODEL_PATH", DENSE_MODEL_PATHS["qwen2_5_vl"])


def write_result(path: str, result: str):
    with open(path, "w") as f:
        f.write(result)


def mock_vlm_input(engine: "MegatronEngine") -> dict[str, Any]:
    """Create mock VLM input with vision tokens.

    Pulls patch geometry from ``engine.hf_config`` so this works for both
    Qwen2.5-VL (patch_size=14) and Qwen3-VL (patch_size=16).
    """
    vc = engine.hf_config.vision_config
    patch_size = getattr(vc, "patch_size", 14)
    temporal_patch_size = getattr(vc, "temporal_patch_size", 2)
    spatial_merge_size = getattr(vc, "spatial_merge_size", 2)
    in_channels = getattr(vc, "in_channels", 3)
    image_token_id = getattr(engine.hf_config, "image_token_id", 151655)
    device = engine.device

    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

    grid_t, grid_h, grid_w = 1, 4, 4
    total_patches = grid_t * grid_h * grid_w
    num_image_tokens = (
        grid_t * (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)
    )

    batch_size = 1
    num_text_tokens = 16
    seq_len = num_text_tokens + num_image_tokens

    text_tokens = torch.randint(0, 1000, (num_text_tokens,), dtype=torch.long)
    image_tokens = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)
    input_ids = torch.cat([text_tokens, image_tokens]).unsqueeze(0).to(device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    pixel_values = torch.randn(
        total_patches, patch_dim, dtype=torch.float32, device=device
    )
    image_grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]], dtype=torch.long, device=device
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "multi_modal_input": [
            {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
        ],
    }


def make_vlm_engine(
    backend: str,
    init_optimizer: bool = False,
    wrap_with_ddp: bool = True,
) -> MegatronEngine:
    """Build a MegatronEngine for VLM tests.

    ``wrap_with_ddp=False`` skips the DDP grad-buffer allocation (~2× model
    bf16 bytes per rank). Forward-only / save-only paths don't need grads
    and benefit from the memory headroom — important for 30B+ MoE models
    where the grad buffer pushes per-GPU usage near 80 GB and tiny NCCL
    coordination allocations can fail when other processes hold a few
    hundred MB on the same device.
    """
    bridge_type = os.environ.get("AREAL_TEST_BRIDGE_TYPE", "mbridge")
    config = TrainEngineConfig(
        backend=backend,
        experiment_name="test-vlm",
        trial_name="test",
        path=VLM_MODEL_PATH,
        mb_spec=MicroBatchSpec(max_tokens_per_mb=4096),
        optimizer=OptimizerConfig() if init_optimizer else None,
        megatron=MegatronEngineConfig(
            bridge_type=bridge_type, wrap_with_ddp=wrap_with_ddp
        ),
        gradient_checkpointing=True,
    )
    alloc_mode = ModelAllocation.from_str(backend)
    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=32, train_batch_size=2)
    engine = MegatronEngine(config)
    engine.create_process_group(parallel_strategy=alloc_mode.parallel)
    engine.initialize(addr=None, ft_spec=ft_spec)
    return engine


def _make_input(engine: MegatronEngine) -> dict[str, Any]:
    """Create mock input and broadcast across model-parallel ranks."""
    input_ = mock_vlm_input(engine)
    return broadcast_tensor_container(
        input_,
        src_rank=engine.current_data_parallel_head(),
        group=engine.context_and_model_parallel_group,
    )


def _cleanup(engine: MegatronEngine):
    torch.cuda.synchronize()
    dist.barrier()
    engine.destroy()


def test_vlm_init(backend: str, output: str | None = None):
    """Test VLM engine initialization: model detection and processor loading."""
    rank = int(os.environ["RANK"])
    engine = make_vlm_engine(backend, init_optimizer=False)

    assert engine.is_vision_model, "Engine should detect VLM model"
    assert engine.processor is not None, "Processor should be loaded for VLM"
    assert engine.tokenizer is not None, "Tokenizer should be loaded"

    _cleanup(engine)
    if rank == 0 and output is not None:
        write_result(output, "Passed")
    print(f"rank {rank}: test_vlm_init({backend}) Done.")


def test_vlm_forward(backend: str, output: str | None = None):
    """Test VLM eval forward pass."""
    rank = int(os.environ["RANK"])
    # No DDP grad buffer — forward is inference-only. Saves ~2× model bytes
    # per rank and avoids NCCL OOMs at the GPU-memory boundary on 30B+ MoE.
    engine = make_vlm_engine(backend, init_optimizer=False, wrap_with_ddp=False)
    bcasted_input = _make_input(engine)

    engine.eval()
    result = engine.forward(bcasted_input)
    assert result is not None, "Forward pass should return a result"

    _cleanup(engine)
    if rank == 0 and output is not None:
        write_result(output, "Passed")
    print(f"rank {rank}: test_vlm_forward({backend}) Done.")


def test_vlm_save_load(backend: str, save_dir: str, output: str | None = None):
    """Test VLM save/load weight round-trip."""
    rank = int(os.environ["RANK"])
    engine = make_vlm_engine(backend, init_optimizer=False)
    bcasted_input = _make_input(engine)

    engine.eval()
    with torch.no_grad():
        old = engine.forward(bcasted_input)

        meta = SaveLoadMeta(
            path=Path(save_dir),
            weight_format="hf",
            tokenizer=engine.tokenizer,
            processor=engine.processor,
            with_optim=False,
            base_model_path=None,
        )
        engine.save(meta)

        if rank == 0:
            has_processor = (Path(save_dir) / "preprocessor_config.json").exists() or (
                Path(save_dir) / "processor_config.json"
            ).exists()
            assert has_processor, "Processor config should be saved"

        for param in engine.model.parameters():
            param.zero_()
        engine.load(meta)

        new = engine.forward(bcasted_input)
        torch.testing.assert_close(old, new)

    _cleanup(engine)
    if rank == 0 and output is not None:
        write_result(output, "Passed")
    print(f"rank {rank}: test_vlm_save_load({backend}) Done.")


def test_vlm_dcp_save_load(backend: str, output: str | None = None):
    """Test VLM DCP save/load round-trip on parameters only.

    Parameter-only round-trip (no forward), so it works for any registered
    VLM regardless of input shape. Mirrors ``test_simple_dcp_save_load`` in
    ``run_megatron_engine_distributed.py`` but uses ``make_vlm_engine`` to
    pull processor/tokenizer through the engine init.
    """
    import copy
    import tempfile

    rank = int(os.environ["RANK"])
    base_dir = tempfile.gettempdir()
    save_path = Path(base_dir) / "megatron_vlm_simple_dcp_test"
    if rank == 0:
        save_path.mkdir(parents=True, exist_ok=True)
    # No barrier here — the process group is not initialized yet;
    # ``make_vlm_engine`` calls ``create_process_group`` + ``initialize``
    # which barrier internally before any rank touches ``save_path``.

    engine = make_vlm_engine(backend, init_optimizer=True)

    meta = SaveLoadMeta(
        path=save_path,
        weight_format="dcp",
        tokenizer=engine.tokenizer,
        with_optim=False,
        base_model_path=None,
    )

    with torch.no_grad():
        engine.eval()
        params = copy.deepcopy(dict(engine.model.named_parameters()))
        engine.save(meta)

        for p in engine.model.parameters():
            p.data.zero_()

        engine.load(meta)

        succ = True
        for name, param in engine.model.named_parameters():
            if not torch.allclose(param, params[name]):
                diff = torch.abs(params[name] - param)
                print(
                    f"rank {rank} diff of {name}: max={torch.max(diff)} "
                    f"avg={torch.mean(diff)} count={torch.count_nonzero(diff)}"
                )
                succ = False
        assert succ, "Weights should be identical after DCP save/load"

    _cleanup(engine)
    if rank == 0 and output is not None:
        write_result(output, "Passed")
    print(f"rank {rank}: test_vlm_dcp_save_load({backend}) Done.")


def test_vlm_train(backend: str, output: str | None = None):
    """Test VLM training step."""
    rank = int(os.environ["RANK"])

    try:
        engine = make_vlm_engine(backend, init_optimizer=True)
        bcasted_input = _make_input(engine)

        engine.train()
        train_result = engine.train_batch(
            input_=bcasted_input,
            loss_fn=lambda logprobs, entropy, input_data, **kwargs: torch.mean(
                logprobs
            ),
            loss_weight_fn=lambda x: torch.tensor(1.0, device=engine.device),
        )

        assert "grad_norm" in train_result, f"Missing grad_norm: {train_result}"
        assert "lr" in train_result, f"Missing lr: {train_result}"
        print(f"rank {rank} train_result: {train_result}")

        _cleanup(engine)
        if rank == 0 and output is not None:
            write_result(output, "Passed")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if rank == 0 and output is not None:
                write_result(output, "OOM")
            print(f"rank {rank}: OOM during training")
            return
        raise

    print(f"rank {rank}: test_vlm_train({backend}) Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="megatron:d1p1t2")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["init", "forward", "save_load", "dcp_save_load", "train"],
        default="train",
    )
    args = parser.parse_args()

    if args.test_type == "init":
        test_vlm_init(args.backend, output=args.output)
    elif args.test_type == "forward":
        test_vlm_forward(args.backend, output=args.output)
    elif args.test_type == "save_load":
        assert args.save_dir is not None, "--save_dir required for save_load test"
        test_vlm_save_load(args.backend, args.save_dir, output=args.output)
    elif args.test_type == "dcp_save_load":
        test_vlm_dcp_save_load(args.backend, output=args.output)
    elif args.test_type == "train":
        test_vlm_train(args.backend, output=args.output)
    else:
        raise NotImplementedError(f"Unknown test type: {args.test_type}")


if __name__ == "__main__":
    main()
