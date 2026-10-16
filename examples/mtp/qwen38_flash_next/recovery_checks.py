# SPDX-License-Identifier: Apache-2.0
"""Record model, optimizer, scheduler and RNG at explicit recovery boundaries."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_state(item) for item in value)
    return value


def state_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            add(torch.from_numpy(item.copy()))
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
            digest.update(b"}")
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                add(child)
            digest.update(b"]")
        else:
            digest.update((type(item).__name__ + ":" + repr(item) + ";").encode())

    add(value)
    return digest.hexdigest()


def frozen_model_state(models: list[torch.nn.Module]) -> dict:
    """Fingerprint all frozen parameters and semantic PLE buffers by ownership."""
    from examples.mtp.qwen38_flash_next.parameter_checks import classify_parameters

    frozen = {
        identity: parameter
        for identity, (_, parameter, is_mtp) in classify_parameters(models).items()
        if not is_mtp
    }
    aliases: dict[int, list[str]] = {}
    buffers: dict[int, torch.Tensor] = {}
    buffer_aliases: dict[int, list[str]] = {}
    ple_buffer_names = {
        "layer_multipliers",
        "ngram_heads_offsets",
        "ngram_heads_vocab_sizes",
    }
    for chunk, model in enumerate(models):
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if id(parameter) in frozen:
                aliases.setdefault(id(parameter), []).append(f"chunk_{chunk}.{name}")
        for name, buffer in model.named_buffers(remove_duplicate=False):
            if (
                "ple" in name.split(".")[:-1]
                and name.split(".")[-1] in ple_buffer_names
            ):
                buffers[id(buffer)] = buffer
                buffer_aliases.setdefault(id(buffer), []).append(
                    f"chunk_{chunk}.{name}"
                )

    def manifest(tensors, names):
        entries = {}
        for identity, tensor in tensors.items():
            registered_names = sorted(names[identity])
            value = tensor.detach()
            if value.ndim == 0:
                value = value.reshape(1)
            row_bytes = value[0].numel() * value.element_size() if value.shape[0] else 1
            rows = max(1, (16 << 20) // max(1, row_bytes))
            digest = hashlib.sha256()
            for start in range(0, value.shape[0], rows):
                block = value[start : start + rows].cpu().contiguous()
                digest.update(block.reshape(-1).view(torch.uint8).numpy().tobytes())
            entries[registered_names[0]] = {
                "aliases": registered_names,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": digest.hexdigest(),
            }
        return entries

    return {
        "parameters": manifest(frozen, aliases),
        "ple_buffers": manifest(buffers, buffer_aliases),
    }


def optimizer_state(optimizer: Any) -> Any:
    if hasattr(optimizer, "chained_optimizers"):
        return [optimizer_state(child) for child in optimizer.chained_optimizers]
    underlying = getattr(optimizer, "optimizer", None)
    if underlying is None:
        return {"state": {}, "master_parameters": []}
    return {
        "state": cpu_state(underlying.state_dict()),
        "master_parameters": [
            [cpu_state(parameter) for parameter in group["params"]]
            for group in underlying.param_groups
        ],
    }


def snapshot_training_state(
    actor: Any, label: str, *, frozen_only: bool = False
) -> None:
    root = os.environ.get("RECOVERY_STATE_DIR")
    if not root:
        return
    from examples.mtp.qwen38_flash_next.parameter_checks import classify_parameters

    state = {"frozen_parameters": frozen_model_state(actor.model)}
    if not frozen_only:
        state.update(
            {
                "parameters": {
                    name: cpu_state(parameter)
                    for name, parameter, is_mtp in classify_parameters(
                        actor.model
                    ).values()
                    if is_mtp
                },
                "optimizer": optimizer_state(actor.optimizer),
                "scheduler": cpu_state(actor.lr_scheduler.state_dict()),
                "rng": cpu_state(actor.checkpointer.get_rng_state(use_dist_ckpt=False)),
            }
        )
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{label}.rank-{dist.get_rank():02d}.pt"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite recovery state {path}")
    torch.save(state, path)


def assert_state_close(
    expected: Any, actual: Any, *, atol: float, rtol: float, path: str = "state"
) -> int:
    """Compare every tensor and metadata leaf, including optimizer moments."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: expected tensor")
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{path}: {message}",
        )
        return 1
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise AssertionError(f"{path}: key mismatch")
        return sum(
            assert_state_close(
                value, actual[key], atol=atol, rtol=rtol, path=f"{path}.{key}"
            )
            for key, value in expected.items()
        )
    if isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: length mismatch")
        return sum(
            assert_state_close(
                left, right, atol=atol, rtol=rtol, path=f"{path}[{index}]"
            )
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
        )
    if expected != actual:
        raise AssertionError(f"{path}: metadata mismatch {expected!r} != {actual!r}")
    return 0
