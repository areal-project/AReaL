# SPDX-License-Identifier: Apache-2.0

"""Full, fixed-layout checkpoint sidecars for unmodified native train engines."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from areal.utils.atomic import atomic_json, recovery_digest

from .runtime_state import (
    capture_runtime_state,
    restore_random_state,
    restore_runtime_state,
)

Backend = Literal["fsdp", "megatron"]


def _tracker() -> Any:
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return get_cuda_rng_tracker()


def _capture(engine: Any, backend: Backend, identity_sha256: str) -> dict[str, Any]:
    state = capture_runtime_state(engine)
    state.update(
        backend=backend,
        identity_sha256=identity_sha256,
        optimizer_group_lrs=[group["lr"] for group in engine.optimizer.param_groups],
    )
    if backend == "megatron":
        state["megatron_rng_tracker"] = {
            name: value.cpu().tolist()
            for name, value in _tracker().get_states().items()
        }
    return state


def _restore_rng(engine: Any, state: dict[str, Any]) -> None:
    restore_random_state(engine, state)
    if state["backend"] == "megatron":
        import torch

        _tracker().set_states(
            {
                name: torch.tensor(value, dtype=torch.uint8)
                for name, value in state["megatron_rng_tracker"].items()
            }
        )


def _report(state: dict[str, Any]) -> dict[str, Any]:
    return {
        **{
            key: state[key]
            for key in (
                "rank",
                "world_size",
                "policy_version",
                "backend",
                "identity_sha256",
            )
        },
        "state_sha256": recovery_digest(state),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _validate_payload(root: Path, backend: Backend) -> None:
    if not (root / "dcp" / ".metadata").is_file():
        raise ValueError("Checkpoint has no finalized DCP metadata")
    if backend == "megatron":
        marker = root / "dcp" / "metadata.json"
        if not marker.is_file() or not (root / "dcp" / "common.pt").is_file():
            raise ValueError("Megatron checkpoint has not finished finalization")
        metadata = json.loads(marker.read_text())
        if (
            metadata.get("sharded_backend"),
            metadata.get("sharded_backend_version"),
            metadata.get("common_backend"),
            metadata.get("common_backend_version"),
        ) != ("torch_dist", 1, "torch", 1):
            raise ValueError("Unsupported Megatron checkpoint serialization format")


def inspect_full_checkpoint(
    engine: Any,
    *,
    path: str,
    backend: Backend,
    identity_sha256: str,
    version: int,
    loading: bool = False,
    restore_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collective-free preflight; every rank must pass before any rank loads."""
    if backend not in ("fsdp", "megatron") or type(version) is not int or version < 1:
        raise ValueError(
            "Full checkpoint requires a supported backend and positive version"
        )
    if len(identity_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in identity_sha256
    ):
        raise ValueError("Checkpoint identity must be a SHA256 digest")
    root = Path(path)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("Checkpoint root must be an existing absolute directory")
    if engine.optimizer is None or engine.lr_scheduler is None:
        raise ValueError("Full recovery requires optimizer and scheduler state")
    if backend == "megatron":
        if (
            engine.checkpointer is None
            or not engine.mcore_config.use_checkpoint_opt_param_scheduler
        ):
            raise ValueError(
                "Megatron requires full distributed optimizer/scheduler checkpoint support"
            )
    if restore_plan is not None and (not loading or backend != "fsdp"):
        raise ValueError(
            "An explicit DP restore plan is only supported for FSDP loading"
        )
    if loading:
        state = _loading_state(
            engine, root, backend, identity_sha256, version, restore_plan
        )
        if (
            state.get("format_version"),
            state.get("rank"),
            state.get("world_size"),
            state.get("policy_version"),
            state.get("backend"),
            state.get("identity_sha256"),
        ) != (1, engine.rank, engine.world_size, version, backend, identity_sha256):
            raise ValueError("Checkpoint runtime identity does not match this worker")
        _validate_payload(root, backend)
        for key in (
            "lr_scheduler",
            "optimizer_group_lrs",
            "python_rng",
            "numpy_rng",
            "torch_rng",
            "cuda_rng",
        ):
            if key not in state:
                raise ValueError(f"Checkpoint is missing {key}")
        if backend == "megatron" and "megatron_rng_tracker" not in state:
            raise ValueError("Checkpoint is missing Megatron RNG tracker state")
        if (engine.device.type == "cuda") != (state["cuda_rng"] is not None):
            raise ValueError("Checkpoint device type does not match this worker")
    else:
        # MCore expects the shared directory to exist before entering any
        # save collectives. Every rank performs this collective-free preflight.
        (root / "dcp").mkdir(exist_ok=True)
        if engine.get_version() != version:
            raise ValueError("Worker version does not match the checkpoint update")
        state = _capture(engine, backend, identity_sha256)
    return _report(state)


def _loading_state(
    engine: Any,
    root: Path,
    backend: Backend,
    identity_sha256: str,
    version: int,
    restore_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if restore_plan is not None:
        from .dp_restore import mapped_runtime_state

        return mapped_runtime_state(
            engine,
            path=root,
            identity_sha256=identity_sha256,
            version=version,
            plan=restore_plan,
        )
    return json.loads((root / f"runtime-{engine.rank}.json").read_text())


def save_full_checkpoint(
    engine: Any,
    *,
    path: str,
    backend: Backend,
    identity_sha256: str,
    version: int,
) -> dict[str, Any]:
    from areal.api.io_struct import SaveLoadMeta

    inspect_full_checkpoint(
        engine,
        path=path,
        backend=backend,
        identity_sha256=identity_sha256,
        version=version,
    )
    state = _capture(engine, backend, identity_sha256)
    engine.save(
        SaveLoadMeta(path=str(Path(path) / "dcp"), weight_format="dcp", with_optim=True)
    )
    if backend == "megatron":
        # The finalize operation includes collectives, even on a rank with no
        # locally pending writes. Never condition it on local queue depth.
        engine.checkpointer.wait_async_saves()
    _validate_payload(Path(path), backend)
    _restore_rng(engine, state)
    atomic_json(Path(path) / f"runtime-{engine.rank}.json", state)
    return _report(state)


def load_full_checkpoint(
    engine: Any,
    *,
    path: str,
    backend: Backend,
    identity_sha256: str,
    version: int,
    restore_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from areal.api.io_struct import SaveLoadMeta

    inspect_full_checkpoint(
        engine,
        path=path,
        backend=backend,
        identity_sha256=identity_sha256,
        version=version,
        loading=True,
        restore_plan=restore_plan,
    )
    state = _loading_state(
        engine, Path(path), backend, identity_sha256, version, restore_plan
    )
    engine.load(
        SaveLoadMeta(path=str(Path(path) / "dcp"), weight_format="dcp", with_optim=True)
    )
    if backend == "fsdp":
        restore_runtime_state(engine, state)
    else:
        # Megatron's DCP loader already restores the scheduler. Its load method
        # advances by the saved num_steps, so applying the sidecar a second time
        # would double that progress. The digest below verifies the DCP restore.
        engine.set_version(version)
    if [group["lr"] for group in engine.optimizer.param_groups] != state[
        "optimizer_group_lrs"
    ]:
        raise ValueError("Restored optimizer learning rates disagree with checkpoint")
    _restore_rng(engine, state)
    # Optional RNG metadata was added without invalidating legacy sidecars.
    # Compare/report exactly the fields that were validated in the preflight.
    captured = _capture(engine, backend, identity_sha256)
    actual = {key: captured[key] for key in state}
    if recovery_digest(actual) != recovery_digest(state):
        different_fields = sorted(
            key
            for key in state.keys() | actual.keys()
            if recovery_digest({key: state.get(key)})
            != recovery_digest({key: actual.get(key)})
        )
        raise ValueError(
            "Full checkpoint runtime state was not restored exactly: "
            + ", ".join(different_fields)
        )
    return _report(actual)
