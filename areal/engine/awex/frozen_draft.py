# SPDX-License-Identifier: Apache-2.0

"""Preserve frozen speculative weights across colocated memory release."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def get_draft_model(scheduler: Any) -> Any | None:
    worker = getattr(scheduler, "draft_worker", None)
    if worker is None:
        return None
    runner = getattr(worker, "draft_model_runner", None)
    if runner is None:
        inner = getattr(worker, "_draft_worker", worker)
        runner = getattr(inner, "draft_runner", None)
    if runner is None:
        raise RuntimeError("Cannot locate frozen draft model for AWEX memory release")
    return runner.model


def _tensors(model: Any) -> dict[str, torch.Tensor]:
    return {
        **{f"parameter:{name}": value for name, value in model.named_parameters()},
        **{f"buffer:{name}": value for name, value in model.named_buffers()},
    }


def _storage(tensor: torch.Tensor) -> tuple[Any, int]:
    return tensor.device, tensor.untyped_storage().data_ptr()


class FrozenDraftState:
    """Keep CPU copies of independent parameters and nonpersistent buffers.

    Native memory-saver backup follows CUDA allocations, which may be shared
    between target and draft regions by the caching allocator. A draft-only
    backup flag therefore does not establish that every draft tensor survives.
    Restore in place before generation resumes to preserve CUDA graph pointers.
    Shared target storage (embedding/head) must receive the new actor weights.
    """

    def __init__(self, target_model: Any, draft_model: Any) -> None:
        self._draft_model = draft_model
        target_storage = {_storage(value) for value in _tensors(target_model).values()}
        self._live = {
            name: value
            for name, value in _tensors(draft_model).items()
            if _storage(value) not in target_storage
        }
        self._cpu = {
            name: value.detach().to(device="cpu", copy=True)
            for name, value in self._live.items()
        }
        self._storage = {name: _storage(value) for name, value in self._live.items()}

    @property
    def tensor_count(self) -> int:
        return len(self._cpu)

    @property
    def nbytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self._cpu.values())

    def restore(self) -> None:
        import torch

        current = _tensors(self._draft_model)
        for name, original in self._live.items():
            value = current.get(name)
            if value is not original or _storage(value) != self._storage[name]:
                raise RuntimeError(f"Frozen draft tensor identity changed: {name}")
        with torch.no_grad():
            for name, value in self._live.items():
                value.copy_(self._cpu[name])
