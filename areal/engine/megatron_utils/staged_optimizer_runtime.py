# SPDX-License-Identifier: Apache-2.0

"""Shared bounded-CUDA runtime for CPU-authoritative optimizers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, TypeVar

import torch


class _SlotPhase(Enum):
    FREE = auto()
    D2H_PENDING = auto()


class SlotStateMachine:
    """Prevent a staging slot from being reused before its D2H completes."""

    def __init__(self, slot_count: int, wait_for_slot: Callable[[int], None]) -> None:
        if slot_count < 1:
            raise ValueError("slot_count must be at least 1")
        self._phases = [_SlotPhase.FREE] * slot_count
        self._wait_for_slot = wait_for_slot

    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(phase.name for phase in self._phases)

    def acquire(self, slot_index: int) -> None:
        if self._phases[slot_index] is _SlotPhase.D2H_PENDING:
            self._wait_for_slot(slot_index)
            self._phases[slot_index] = _SlotPhase.FREE

    def mark_d2h_pending(self, slot_index: int) -> None:
        if self._phases[slot_index] is not _SlotPhase.FREE:
            raise RuntimeError(f"staging slot {slot_index} is already in use")
        self._phases[slot_index] = _SlotPhase.D2H_PENDING

    def drain(self) -> None:
        for slot_index, phase in enumerate(self._phases):
            if phase is _SlotPhase.D2H_PENDING:
                self._wait_for_slot(slot_index)
                self._phases[slot_index] = _SlotPhase.FREE


@dataclass
class CUDAStagingSlot:
    """One reusable set of FP32 buffers and its three-stream pipeline."""

    buffers: dict[str, torch.Tensor]
    h2d_stream: torch.cuda.Stream
    compute_stream: torch.cuda.Stream
    d2h_stream: torch.cuda.Stream
    h2d_done: torch.cuda.Event
    compute_done: torch.cuda.Event
    d2h_done: torch.cuda.Event

    @classmethod
    def allocate(
        cls,
        capacity: int,
        device: torch.device,
        buffer_names: Sequence[str],
    ) -> CUDAStagingSlot:
        if capacity < 1:
            raise ValueError("staging slot capacity must be positive")
        tensor_kwargs = {"size": (capacity,), "dtype": torch.float32, "device": device}
        return cls(
            buffers={name: torch.empty(**tensor_kwargs) for name in buffer_names},
            h2d_stream=torch.cuda.Stream(device=device),
            compute_stream=torch.cuda.Stream(device=device),
            d2h_stream=torch.cuda.Stream(device=device),
            h2d_done=torch.cuda.Event(),
            compute_done=torch.cuda.Event(),
            d2h_done=torch.cuda.Event(),
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return self.buffers[name]
        except KeyError as error:
            raise AttributeError(name) from error


UnitT = TypeVar("UnitT")


class StagedOptimizerRuntime:
    """Own shared slot allocation, scheduling, and residency transitions."""

    def __init__(self, buffer_count: int, buffer_names: Sequence[str]) -> None:
        if buffer_count < 1:
            raise ValueError("buffer_count must be at least 1")
        if not buffer_names or len(set(buffer_names)) != len(buffer_names):
            raise ValueError("staging buffer names must be non-empty and unique")
        self._buffer_count = buffer_count
        self._buffer_names = tuple(buffer_names)
        self.slots: list[CUDAStagingSlot] = []
        self.slot_machine: SlotStateMachine | None = None
        self.residency = "UNBOUND"
        self.checkpoint_load_error: BaseException | None = None
        self._bound = False

    def bind(self, *, capacity: int | None, device: torch.device | None) -> None:
        if self._bound:
            raise RuntimeError("staging runtime is already bound")
        self._bound = True
        if capacity is None:
            self.residency = "CPU_RESIDENT"
            return
        if device is None or device.type != "cuda":
            raise ValueError("staging slots require a CUDA device")
        self._allocate_slots(capacity, device)
        self.residency = "STEP_ACTIVE"

    def _allocate_slots(self, capacity: int, device: torch.device) -> None:
        if self.slots or self.slot_machine is not None:
            raise RuntimeError("staging slots are already allocated")
        self.slots.extend(
            CUDAStagingSlot.allocate(capacity, device, self._buffer_names)
            for _ in range(self._buffer_count)
        )
        self.slot_machine = SlotStateMachine(len(self.slots), self._wait_for_slot)

    def _wait_for_slot(self, slot_index: int) -> None:
        self.slots[slot_index].d2h_done.synchronize()

    def acquire_slot(self, slot_index: int) -> CUDAStagingSlot:
        if self.slot_machine is None:
            raise RuntimeError("staging slots are not allocated")
        self.slot_machine.acquire(slot_index)
        return self.slots[slot_index]

    def mark_d2h_pending(self, slot_index: int) -> None:
        if self.slot_machine is None:
            raise RuntimeError("staging slots are not allocated")
        self.slot_machine.mark_d2h_pending(slot_index)

    def schedule_units(
        self,
        units: Sequence[UnitT],
        schedule: Callable[[UnitT, int, torch.cuda.Event], None],
        *,
        wait_for_compute: bool,
    ) -> None:
        if not units:
            self.residency = "CPU_RESIDENT"
            return
        if not self.slots or self.slot_machine is None:
            raise RuntimeError("staging slots are not allocated")
        device = self.slots[0].buffers[self._buffer_names[0]].device
        caller_stream = torch.cuda.current_stream(device)
        inputs_ready = torch.cuda.Event()
        inputs_ready.record(caller_stream)
        self.residency = "STEP_ACTIVE"
        for unit_index, unit in enumerate(units):
            schedule(unit, unit_index % len(self.slots), inputs_ready)
        if wait_for_compute:
            for slot_index, phase in enumerate(self.slot_machine.phases):
                if phase == _SlotPhase.D2H_PENDING.name:
                    caller_stream.wait_event(self.slots[slot_index].compute_done)

    def drain(self) -> None:
        if self.slot_machine is not None:
            self.slot_machine.drain()
        if self._bound:
            self.residency = "CPU_RESIDENT"

    def release_slots(self) -> None:
        self.drain()
        self.slots.clear()
        self.slot_machine = None

    def restore_slots(self, *, capacity: int, device: torch.device) -> None:
        if not self._bound or self.slots:
            return
        if device.type != "cuda":
            raise ValueError("staging slots require a CUDA device")
        self._allocate_slots(capacity, device)
        self.residency = "CPU_RESIDENT"

    def mark_checkpoint_load_failed(self, error: BaseException) -> None:
        self.checkpoint_load_error = error
        self.residency = "CPU_RESIDENT"
