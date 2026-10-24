# SPDX-License-Identifier: Apache-2.0

"""Per-rank scheduler, version and RNG state for native training checkpoints."""

import random
import sys
from typing import Any


def capture_runtime_state(engine: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    np_state = np.random.get_state()
    seeding = sys.modules.get("areal.utils.seeding")
    tracker = getattr(
        sys.modules.get("torch.distributed.tensor._random"), "_rng_tracker", None
    )
    return {
        "format_version": 1,
        "rank": engine.rank,
        "world_size": engine.world_size,
        "policy_version": engine.get_version(),
        "lr_scheduler": engine.lr_scheduler.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch_rng": torch.get_rng_state().tolist(),
        "cuda_rng": (
            torch.cuda.get_rng_state(engine.device).cpu().tolist()
            if engine.device.type == "cuda"
            else None
        ),
        "seed_state": None
        if seeding is None
        else {
            "seed": seeding._SEED,
            "base_seed": seeding._BASE_SEED,
            "shuffler": None
            if seeding._SHUFFLER is None
            else vars(seeding._SHUFFLER).copy(),
        },
        "dtensor_rng": None
        if engine.device.type != "cuda" or not hasattr(engine, "parallel_helper")
        else {
            "initialized": tracker is not None,
            "distribute_region_enabled": None
            if tracker is None
            else tracker.distribute_region_enabled,
        },
    }


def restore_runtime_state(engine: Any, state: dict[str, Any]) -> None:
    if state["format_version"] != 1:
        raise ValueError("Unsupported runtime state format")
    if (state["rank"], state["world_size"]) != (engine.rank, engine.world_size):
        raise ValueError("Runtime state rank/world size mismatch")
    needs_cuda = engine.device.type == "cuda"
    if needs_cuda != (state["cuda_rng"] is not None):
        raise ValueError("Runtime state device type mismatch")
    engine.lr_scheduler.load_state_dict(state["lr_scheduler"])
    expected_lrs = engine.lr_scheduler.get_last_lr()
    actual_lrs = [g["lr"] for g in engine.optimizer.param_groups]
    if actual_lrs != expected_lrs:
        raise ValueError(
            f"Optimizer/scheduler learning rates disagree: {actual_lrs} != {expected_lrs}"
        )
    engine.set_version(state["policy_version"])
    restore_random_state(engine, state)


def restore_random_state(engine: Any, state: dict[str, Any]) -> None:
    """Restore per-rank RNG without assuming a backend's scheduler interface."""
    import numpy as np
    import torch

    if (state["rank"], state["world_size"]) != (engine.rank, engine.world_size):
        raise ValueError("Runtime state rank/world size mismatch")
    needs_cuda = engine.device.type == "cuda"
    if needs_cuda != (state["cuda_rng"] is not None):
        raise ValueError("Runtime state device type mismatch")
    metadata = state.get("dtensor_rng")
    if metadata is not None:
        from torch.distributed.tensor import _random as dtensor_rng

        if not metadata["initialized"]:
            dtensor_rng._rng_tracker = None
        else:
            # Torch 2.9/2.10: avoid the lazy tracker's WORLD-rank-zero broadcast
            # overwriting the restored rank-local CUDA stream on its first use.
            if dtensor_rng._rng_tracker is None:
                dtensor_rng._rng_tracker = dtensor_rng.OffsetBasedRNGTracker(
                    engine.parallel_helper.world_mesh["tp"], run_state_sync=False
                )
            dtensor_rng._rng_tracker.distribute_region_enabled = metadata[
                "distribute_region_enabled"
            ]
    py_state = state["python_rng"]
    random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
    np_state = state["numpy_rng"]
    np.random.set_state(
        (np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:])
    )
    torch.set_rng_state(torch.tensor(state["torch_rng"], dtype=torch.uint8))
    if needs_cuda:
        torch.cuda.set_rng_state(
            torch.tensor(state["cuda_rng"], dtype=torch.uint8), engine.device
        )
    if state.get("seed_state") is not None:
        from areal.utils import seeding

        metadata = state["seed_state"]
        seeding._SEED, seeding._BASE_SEED = metadata["seed"], metadata["base_seed"]
        seeding._SHUFFLER = None
        if metadata["shuffler"] is not None:
            seeding._SHUFFLER = seeding.Shuffler()
            vars(seeding._SHUFFLER).update(metadata["shuffler"])


def reseed_runtime_state(
    engine: Any, source: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Build target-rank streams without modifying global RNGs during preflight."""
    from copy import deepcopy

    import numpy as np
    import torch

    state = deepcopy(source)
    np_state = np.random.RandomState(seed).get_state()
    state.update(
        rank=engine.rank,
        world_size=engine.world_size,
        python_rng=random.Random(seed).getstate(),
        numpy_rng=[np_state[0], np_state[1].tolist(), *np_state[2:]],
        torch_rng=torch.Generator(device="cpu").manual_seed(seed).get_state().tolist(),
        cuda_rng=None
        if engine.device.type != "cuda"
        else torch.Generator(device=engine.device)
        .manual_seed(seed)
        .get_state()
        .cpu()
        .tolist(),
    )
    if "seed_state" in state and state["seed_state"] is not None:
        state["seed_state"] = {"seed": seed, "base_seed": seed, "shuffler": None}
    if "dtensor_rng" in state and state["dtensor_rng"] is not None:
        state["dtensor_rng"] = {"initialized": True, "distribute_region_enabled": True}
    return state
