# SPDX-License-Identifier: Apache-2.0

"""Release activation storage retained by Megatron and Transformer Engine."""

from __future__ import annotations

import sys
import weakref
from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch

_tracking_installed = False
_active_tracker: ActivationStorageTracker | None = None


class ActivationStorageTracker:
    """Own activation tensors retained within one training forward/backward."""

    def __init__(
        self,
        protected_tensors: Iterable[torch.Tensor] = (),
        *,
        strong_references: bool = False,
    ) -> None:
        self._tensor_refs: deque[torch.Tensor | weakref.ReferenceType[torch.Tensor]] = (
            deque()
        )
        self._strong_references = strong_references
        self._protected_storages = {
            tensor.untyped_storage().data_ptr()
            for tensor in protected_tensors
            if isinstance(tensor, torch.Tensor)
            and tensor.is_cuda
            and tensor.untyped_storage().nbytes() > 0
        }
        self._nested_released_storages = 0
        self._nested_released_bytes = 0

    def child(
        self, protected_tensors: Iterable[torch.Tensor] = ()
    ) -> ActivationStorageTracker:
        """Create a nested tracker that inherits protected storage ownership."""
        child = ActivationStorageTracker(protected_tensors, strong_references=True)
        child._protected_storages.update(self._protected_storages)
        return child

    def record_nested_release(self, storage_count: int, released_bytes: int) -> None:
        self._nested_released_storages += storage_count
        self._nested_released_bytes += released_bytes

    def protect(self, tensors: Iterable[torch.Tensor]) -> None:
        """Exclude storages whose ownership leaves this tracker scope."""
        self._protected_storages.update(
            tensor.untyped_storage().data_ptr()
            for tensor in tensors
            if isinstance(tensor, torch.Tensor)
            and tensor.is_cuda
            and tensor.untyped_storage().nbytes() > 0
        )

    def append(self, tensor: torch.Tensor) -> None:
        if not tensor.is_cuda:
            return
        if tensor.untyped_storage().data_ptr() in self._protected_storages:
            return
        if self._strong_references:
            self._tensor_refs.append(tensor)
        else:
            self._tensor_refs.append(weakref.ref(tensor))

    @torch.no_grad()
    def release(self) -> tuple[int, int]:
        """Release storage after the complete backward has returned."""
        released_storages: set[int] = set()
        released_bytes = 0

        while self._tensor_refs:
            tensor_ref = self._tensor_refs.popleft()
            tensor = (
                tensor_ref()
                if isinstance(tensor_ref, weakref.ReferenceType)
                else tensor_ref
            )
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            data_ptr = storage.data_ptr()
            nbytes = storage.nbytes()
            if data_ptr in self._protected_storages:
                continue
            if nbytes > 0 and data_ptr not in released_storages:
                released_storages.add(data_ptr)
                released_bytes += nbytes
                storage.resize_(0)
            if tensor._base is None:
                tensor.detach_()

        return (
            len(released_storages) + self._nested_released_storages,
            released_bytes + self._nested_released_bytes,
        )

    def discard(self) -> None:
        """Drop Python references without mutating an incomplete autograd graph."""
        self._tensor_refs.clear()


def install_activation_storage_tracking() -> None:
    """Patch Megatron and TE activation boundaries for scoped storage tracking.

    Megatron modules commonly import ``make_viewless_tensor`` into module-local
    aliases. Replace both the canonical function and aliases already loaded so
    the tracker records the final tensor returned by ``Function.apply``.

    Transformer Engine's operation fuser marks activation inputs and outputs as
    non-clearable during backward. Some versions leave the corresponding C++
    autograd references alive beyond the backward call. Selective recompute also
    builds a nested autograd graph inside ``CheckpointFunction.backward``. Track
    normal forward boundaries weakly so tracking does not extend their lifetime.
    During checkpoint recompute, hold boundaries strongly only for the lifetime
    of the nested backward and release them as soon as its checkpoint returns.
    """
    global _tracking_installed

    if _tracking_installed:
        return

    from megatron.core import utils as megatron_utils

    original_make_viewless_tensor = megatron_utils.make_viewless_tensor

    def tracked_make_viewless_tensor(
        inp: torch.Tensor, requires_grad: bool, keep_graph: bool
    ) -> torch.Tensor:
        output = original_make_viewless_tensor(inp, requires_grad, keep_graph)
        tracker = _active_tracker
        if tracker is not None and keep_graph and inp._base is not None:
            # Megatron's viewless tensor aliases ``inp`` storage. Track that
            # shared activation storage and release it only after its owning
            # backward scope completes; cloning here would retain one full
            # hidden-state allocation per layer and inflate the forward peak.
            tracker.append(output)
        return output

    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("megatron.") or module is None:
            continue
        if (
            getattr(module, "make_viewless_tensor", None)
            is original_make_viewless_tensor
        ):
            module.make_viewless_tensor = tracked_make_viewless_tensor

    from transformer_engine.pytorch.ops import fuser as te_fuser

    original_fuser_call = te_fuser.OperationFuser.__call__

    def tracked_fuser_call(
        fuser: te_fuser.OperationFuser,
        input_: torch.Tensor,
        *extra_inputs: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        output = original_fuser_call(fuser, input_, *extra_inputs, **kwargs)
        tracker = _active_tracker
        if tracker is not None:
            tracker.append(input_)
            if isinstance(output, torch.Tensor):
                tracker.append(output)
            else:
                for tensor in output:
                    tracker.append(tensor)
        return output

    te_fuser.OperationFuser.__call__ = tracked_fuser_call

    from megatron.core.tensor_parallel import random as tensor_parallel_random

    original_checkpoint_backward = tensor_parallel_random.CheckpointFunction.backward

    def tracked_checkpoint_backward(ctx: object, *grad_outputs: torch.Tensor):
        global _active_tracker

        tracker = _active_tracker
        if tracker is None:
            return original_checkpoint_backward(ctx, *grad_outputs)

        # A checkpoint recompute builds a complete nested graph during its own
        # backward. Keep its activation ownership local so each checkpoint can
        # release immediately instead of retaining every layer until the full
        # model backward returns. Checkpoint inputs and output gradients belong
        # to the outer graph and must never have their shared storage resized.
        checkpoint_inputs = getattr(ctx, "saved_tensors", ())
        nested_tracker = tracker.child((*checkpoint_inputs, *grad_outputs))

        def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
            if not tensor.is_leaf:
                nested_tracker.append(tensor)
            return tensor

        def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        _active_tracker = nested_tracker
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
                result = original_checkpoint_backward(ctx, *grad_outputs)
        except BaseException:
            nested_tracker.discard()
            raise
        finally:
            _active_tracker = tracker

        nested_tracker.protect(
            tensor for tensor in result if isinstance(tensor, torch.Tensor)
        )
        released_storages, released_bytes = nested_tracker.release()
        tracker.record_nested_release(released_storages, released_bytes)
        return result

    tensor_parallel_random.CheckpointFunction.backward = staticmethod(
        tracked_checkpoint_backward
    )
    _tracking_installed = True


@contextmanager
def track_activation_storage(
    protected_tensors: Iterable[torch.Tensor] = (),
) -> Iterator[ActivationStorageTracker]:
    """Scope activation storage tracking to one training forward/backward."""
    global _active_tracker

    if _active_tracker is not None:
        raise RuntimeError("Megatron activation storage tracking is already active")
    tracker = ActivationStorageTracker(protected_tensors)
    _active_tracker = tracker
    try:
        yield tracker
    except BaseException:
        tracker.discard()
        raise
    finally:
        _active_tracker = None
