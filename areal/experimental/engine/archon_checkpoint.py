# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import struct
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from areal.utils.logging import getLogger

if TYPE_CHECKING:
    from transformers import AutoProcessor, PreTrainedTokenizerFast

    from areal.experimental.engine.archon_engine import ArchonEngine
    from areal.utils.async_checkpoint import AsyncCheckpointManager

logger = getLogger("ArchonCheckpoint")


# NOTE: Upgrading PyTorch may resolve this in the future.
def _consolidate_shards_distributed(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 8,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """Distribute safetensors consolidation across ranks, with correct PG barrier.

    This replaces ``consolidate_safetensors_files_on_every_rank`` which has a bug:
    its internal ``dist.barrier()`` ignores the *process_group* parameter and uses
    the default (NCCL) PG instead.  When the bg consolidation thread calls that
    NCCL barrier concurrently with the main thread's NCCL collectives, different
    ranks may enqueue the collectives in different order, causing a deadlock.

    """
    from torch.distributed.checkpoint._consolidate_hf_safetensors import (
        _consolidate_safetensors_files,
    )
    from torch.distributed.checkpoint._hf_utils import _gen_file_name

    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)

    unique_indices = set(fqn_to_index_mapping.values())

    # Simple round-robin: index % world_size == rank
    indices_for_this_rank = [idx for idx in unique_indices if idx % world_size == rank]

    filtered_mapping = {
        fqn: idx
        for fqn, idx in fqn_to_index_mapping.items()
        if idx in indices_for_this_rank
    }

    if filtered_mapping:
        max_index = max(unique_indices)
        filtered_filename_mapping = {
            fqn: _gen_file_name(idx, max_index) for fqn, idx in filtered_mapping.items()
        }
        _consolidate_safetensors_files(
            input_dir=input_dir,
            output_dir=output_dir,
            fqn_to_file_mapping=filtered_filename_mapping,
            num_threads=num_threads,
        )

    dist.barrier(group=process_group)


class DCPState(Stateful):
    """DCP wrapper for archon models.

    Key design decisions:
    - Uses flatten_optimizer_state_dict=True to avoid param_group index collisions
      (without flatten, each optimizer uses indices 0, 1, 2... which collide across
      PP stages; with flatten, keys become parameter FQNs which are unique)
    - For PP (len(model_parts) > 1): uses strict=False when loading because each
      PP stage only has subset of keys
    - For non-PP (len(model_parts) == 1): uses strict=True to catch real issues
    """

    def __init__(
        self,
        model_parts: list[nn.Module] | nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        """Initialize DCPState.

        Args:
            model_parts: Single model or list of model parts from pipeline_llm
            optimizer: Optimizer for the model(s)
        """
        if isinstance(model_parts, nn.Module):
            self.model_parts = [model_parts]
        else:
            self.model_parts = model_parts
        self.optimizer = optimizer
        # PP mode uses non-strict loading since each stage only has subset of keys
        self._is_pp = len(self.model_parts) > 1

    def state_dict(self) -> dict[str, Any]:
        """Get state dict for model parts and optimizer using DCP utilities."""
        # Merge model state dicts from all parts
        # cpu_offload=True ensures tensors are on CPU for DCP filesystem writer
        model_state: dict[str, Any] = {}
        model_options = StateDictOptions(cpu_offload=True)
        for model_part in self.model_parts:
            part_state = get_model_state_dict(model_part, options=model_options)
            model_state.update(part_state)

        state: dict[str, Any] = {"model": model_state}

        if self.optimizer is not None:
            optim_options = StateDictOptions(
                flatten_optimizer_state_dict=True,
                cpu_offload=True,
            )

            # Get optimizer state for each model part and merge
            optim_state: dict[str, Any] = {}
            for model_part in self.model_parts:
                part_optim = get_optimizer_state_dict(
                    model_part, self.optimizer, options=optim_options
                )
                optim_state.update(part_optim)

            state["optim"] = optim_state

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dicts onto model parts and optimizer."""
        model_state = state_dict["model"]

        model_options = StateDictOptions(strict=not self._is_pp)
        for model_part in self.model_parts:
            set_model_state_dict(model_part, model_state, options=model_options)

        if self.optimizer is not None and "optim" in state_dict:
            optim_state = state_dict["optim"]
            optim_options = StateDictOptions(
                strict=not self._is_pp,
                flatten_optimizer_state_dict=True,
            )
            for model_part in self.model_parts:
                set_optimizer_state_dict(
                    model_part, self.optimizer, optim_state, options=optim_options
                )


def _validate_model_initialized(engine: ArchonEngine) -> None:
    """Validate that model is properly initialized for checkpoint operations."""
    if not engine.model_parts:
        raise RuntimeError("Model parts not initialized")


def _get_merged_state_dict(
    engine: ArchonEngine,
    options: StateDictOptions,
) -> dict[str, Any]:
    """Get merged model state dict, handling PP mode."""
    if engine.parallel_dims.pp_enabled:
        state_dict: dict = {}
        for model_part in engine.model_parts:
            part_state = get_model_state_dict(model_part, options=options)
            state_dict.update(part_state)
        return state_dict
    return get_model_state_dict(engine.model, options=options)


def _write_safetensors_index(
    output_dir: str, fqn_to_index_mapping: dict[str, int]
) -> None:
    """Write model.safetensors.index.json for multi-file HF checkpoints."""
    max_index = max(fqn_to_index_mapping.values())
    weight_map = {
        fqn: f"model-{idx:05d}-of-{max_index:05d}.safetensors"
        for fqn, idx in fqn_to_index_mapping.items()
    }

    # Compute total_size from safetensors file headers (no tensor loading)
    total_size = 0
    for filename in set(weight_map.values()):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "rb") as f:
            # safetensors format: 8-byte LE header size, then JSON header
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            total_size += end - start

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)


def save_model_to_hf(
    engine: ArchonEngine,
    path: str,
    tokenizer: PreTrainedTokenizerFast | None,
    processor: AutoProcessor | None = None,
    async_mgr: AsyncCheckpointManager | None = None,
) -> None:
    """Save model in HuggingFace format using DCP infrastructure.

    Args:
        engine: The ArchonEngine instance.
        path: Output directory for the HF checkpoint.
        tokenizer: Optional tokenizer to save alongside the model.
        processor: Optional processor to save alongside the model.
        async_mgr: Optional async checkpoint manager. When provided and async
            is enabled, dcp.async_save() is used instead of dcp.save().
            The manager's post_upload_fn is set to handle consolidation.
    """
    from torch.distributed.checkpoint import HuggingFaceStorageWriter

    _validate_model_initialized(engine)
    if engine.state_dict_adapter is None:
        raise RuntimeError("state_dict_adapter is required for HF format")

    engine.logger.info(f"Saving HF checkpoint to {path}")

    # In async mode, let the stager handle GPU->CPU transfer
    is_async = async_mgr is not None and async_mgr.is_async

    # Write to temp dir first, then atomically rename to final path.
    tmp_path = path + ".tmp"
    if dist.get_rank() == 0 and os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    dist.barrier(group=engine.cpu_group)
    os.makedirs(tmp_path, exist_ok=True)
    options = StateDictOptions(full_state_dict=False, cpu_offload=not is_async)
    state_dict = _get_merged_state_dict(engine, options)

    hf_state_dict = engine.state_dict_adapter.to_hf(state_dict)
    fqn_to_index_mapping = engine.state_dict_adapter.fqn_to_index_mapping

    # HuggingFaceStorageWriter creates a sharded/ subdir we must clean up after consolidation.
    sharded_dir = os.path.join(tmp_path, "sharded")

    consolidation_mapping = fqn_to_index_mapping or dict.fromkeys(
        hf_state_dict.keys(), 1
    )

    hf_writer = HuggingFaceStorageWriter(
        path=sharded_dir,
        save_distributed=True,
        fqn_to_index_mapping=fqn_to_index_mapping,
        enable_consolidation=False,
    )

    consolidation_pg = (
        async_mgr.consolidation_process_group if is_async else engine.cpu_group
    )

    def _consolidate_and_cleanup(process_group=consolidation_pg):
        try:
            _consolidate_shards_distributed(
                input_dir=sharded_dir,
                output_dir=tmp_path,
                fqn_to_index_mapping=consolidation_mapping,
                num_threads=8,
                process_group=process_group,
            )
        except Exception:
            # Must re-raise: this function contains a collective barrier, so
            # swallowing the exception on a subset of ranks causes deadlock.
            logger.error("Consolidation failed, keeping sharded dir", exc_info=True)
            raise
        if dist.get_rank(group=process_group) == 0:
            # _consolidate_shards_distributed does not write the
            # index JSON that HuggingFace from_pretrained() needs.
            # Always write it - consolidation_mapping is defined for
            # both multi-file and single-file (fallback) cases.
            _write_safetensors_index(tmp_path, consolidation_mapping)
            if os.path.exists(sharded_dir):
                shutil.rmtree(sharded_dir)
            # Write config / tokenizer / processor into temp dir
            engine.model_config.save_pretrained(tmp_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmp_path)
            if processor is not None:
                processor.save_pretrained(tmp_path)
            # Atomically swap temp dir to final path
            if os.path.exists(path):
                shutil.rmtree(path)
            os.rename(tmp_path, path)
        dist.barrier(group=process_group)

    if is_async:
        async_mgr.save(
            state_dict=hf_state_dict,
            storage_writer=hf_writer,
            post_fn=_consolidate_and_cleanup,
        )
    else:
        dcp.save(hf_state_dict, storage_writer=hf_writer)
        _consolidate_and_cleanup()

    if not is_async:
        dist.barrier(group=engine.cpu_group)


def _check_fp8_shard_compatibility(
    hf_state_dict: dict[str, torch.Tensor],
    scale_keys: list[str],
) -> None:
    """Fail fast if any FP8 weight has non-Shard(0) DTensor placement.

    Must be called before ``_prepare_fp8_state_dict`` / ``dcp.load`` to
    avoid wasting DCP I/O on a configuration that will fail at dequant.
    """
    try:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.placement_types import Shard
    except ImportError:
        return

    for scale_key in scale_keys:
        weight_key = scale_key.replace("_scale_inv", "")
        weight = hf_state_dict.get(weight_key)
        if weight is None or not isinstance(weight, DTensor):
            continue
        for p in weight.placements:
            if isinstance(p, Shard) and p.dim != 0:
                raise ValueError(
                    f"FP8 checkpoint loading does not yet support "
                    f"column-sharded weights (TP/ETP). Weight "
                    f"{weight_key!r} has placements {weight.placements}. "
                    f"Use TP=1 for FP8 checkpoint loading, or wait for "
                    f"Shard(1) dequantization support (Phase 2)."
                )


def load_model_from_hf(engine: ArchonEngine, path: str) -> None:
    """Load model from HuggingFace format using DCP infrastructure."""
    _validate_model_initialized(engine)
    if engine.state_dict_adapter is None:
        raise RuntimeError("state_dict_adapter is required for HF format")

    engine.logger.info(f"Loading HF checkpoint from {path}")

    from areal.experimental.models.archon.fp8_checkpoint import (
        _get_scale_inv_keys,
        _prepare_fp8_state_dict,
        dequant_fp8_state_dict,
    )

    _fp8_scale_keys = _get_scale_inv_keys(path)
    _is_fp8_ckpt = len(_fp8_scale_keys) > 0

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state_dict = _get_merged_state_dict(engine, options)

    # Convert to HF format to match checkpoint keys
    hf_state_dict = engine.state_dict_adapter.to_hf(state_dict)

    # PP mode + weight tying fix: last stage needs embed_tokens weight for output layer
    # When tie_word_embeddings=True, HF checkpoint only stores embed_tokens.weight,
    # not lm_head.weight. In PP mode, last stage has output.weight but no tok_embeddings,
    # so we need to explicitly load embed_tokens.weight even though it's not in state_dict.
    pp_weight_tying_fix = (
        engine.parallel_dims.pp_enabled
        and engine.pp_has_last_stage
        and getattr(engine.state_dict_adapter, "enable_weight_tying", False)
        and "output.weight" in state_dict
    )
    if pp_weight_tying_fix:
        # Add a placeholder with embed_tokens key so DCP will load it
        embed_key = engine.state_dict_adapter._maybe_composite_hf_key(
            "model.embed_tokens.weight"
        )
        if embed_key not in hf_state_dict:
            hf_state_dict[embed_key] = torch.empty_like(state_dict["output.weight"])

    if _is_fp8_ckpt:
        _check_fp8_shard_compatibility(hf_state_dict, _fp8_scale_keys)
        hf_state_dict = _prepare_fp8_state_dict(
            hf_state_dict, path, _cached_keys=_fp8_scale_keys
        )

    # Load using DCP with HuggingFaceStorageReader
    dcp.load(
        hf_state_dict,
        storage_reader=engine.state_dict_adapter.get_hf_storage_reader(path),
    )

    if _is_fp8_ckpt:
        hf_state_dict = dequant_fp8_state_dict(
            hf_state_dict,
            target_dtype=getattr(torch, engine.config.dtype),
        )

    # Convert back to Archon format
    archon_state_dict = engine.state_dict_adapter.from_hf(hf_state_dict)

    # In PP mode, filter to only keep keys needed by this rank's model_parts
    model_keys = set(state_dict.keys())
    if engine.parallel_dims.pp_enabled:
        archon_state_dict = {
            k: v for k, v in archon_state_dict.items() if k in model_keys
        }
    loaded_keys = set(archon_state_dict.keys())

    # Compute key differences for diagnostics
    missing_keys = model_keys - loaded_keys
    unexpected_keys = loaded_keys - model_keys

    # Filter known expected missing keys
    expected_missing = set()
    for key in list(missing_keys):
        # rotary_emb is computed at runtime, not stored in checkpoint
        if "rotary_emb" in key:
            expected_missing.add(key)
    missing_keys -= expected_missing

    if dist.get_rank() == 0:
        if missing_keys:
            engine.logger.warning(
                f"Unexpected missing keys in checkpoint: {missing_keys}"
            )
        if unexpected_keys:
            engine.logger.warning(
                f"Unexpected extra keys in checkpoint: {unexpected_keys}"
            )

    load_options = StateDictOptions(strict=False, full_state_dict=False)
    if engine.parallel_dims.pp_enabled:
        for model_part in engine.model_parts:
            set_model_state_dict(
                model_part,
                model_state_dict=archon_state_dict,
                options=load_options,
            )
    else:
        set_model_state_dict(
            engine.model,
            model_state_dict=archon_state_dict,
            options=load_options,
        )

    dist.barrier(group=engine.cpu_group)


def save_to_dcp(engine: ArchonEngine, path: str, with_optim: bool) -> None:
    """Save model (and optionally optimizer) using DCP format."""
    _validate_model_initialized(engine)

    os.makedirs(path, exist_ok=True)

    dcp_state = DCPState(engine.model_parts, engine.optimizer if with_optim else None)

    state_dict = {"dcp": dcp_state}
    dcp.save(state_dict, checkpoint_id=path)

    dist.barrier(group=engine.cpu_group)


def load_from_dcp(engine: ArchonEngine, path: str, with_optim: bool) -> None:
    """Load model (and optionally optimizer) from DCP format."""
    _validate_model_initialized(engine)

    dcp_state = DCPState(engine.model_parts, engine.optimizer if with_optim else None)

    state_dict = {"dcp": dcp_state}
    dcp.load(state_dict=state_dict, checkpoint_id=path)

    dist.barrier(group=engine.cpu_group)


def save_optimizer_state(engine: ArchonEngine, path: str) -> None:
    """Save optimizer state to disk (sharded by rank)."""
    assert engine.optimizer is not None
    assert dist.is_initialized()
    rank = dist.get_rank()
    shard_path = os.path.join(
        path, f"optim_world_size_{engine.world_size}_rank_{rank}.pt"
    )
    state_dict = engine.optimizer.state_dict()
    torch.save(state_dict, shard_path)
    dist.barrier(group=engine.cpu_group)


def load_optimizer_state(engine: ArchonEngine, path: str) -> None:
    """Load optimizer state from disk (sharded by rank)."""
    assert engine.optimizer is not None
    assert dist.is_initialized()
    rank = dist.get_rank()
    shard_path = os.path.join(
        path, f"optim_world_size_{engine.world_size}_rank_{rank}.pt"
    )
    optimizer_state_dict = torch.load(shard_path, weights_only=False)
    engine.optimizer.load_state_dict(optimizer_state_dict)
    dist.barrier(group=engine.cpu_group)
