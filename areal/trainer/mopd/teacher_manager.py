# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol

from areal.api import SaveLoadMeta
from areal.api.cli_args import MOPDConfig


class TeacherController(Protocol):
    def compute_logp_padded(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[Any] | None, list[Any]]: ...

    def assert_mopd_runtime_topology(self) -> None: ...

    def load(self, meta: SaveLoadMeta) -> None: ...

    def onload(self) -> None: ...

    def offload(self) -> None: ...

    def strict_clear_batches(self, *targets: Any) -> dict[str, int | bool]: ...

    def destroy(self) -> None: ...


@dataclass(frozen=True)
class DrainReceipt:
    """Proof that actor-owned targets no longer reference teacher storage."""

    complete: bool
    source_shards_cleared: int = 0
    actor_fetch_buffers_cleared: int = 0


class TeacherManager(Protocol):
    def pre_fetch(self, teacher_id: str) -> None: ...

    def load(self, teacher_id: str) -> TeacherController: ...

    def release(self, receipt: DrainReceipt) -> None: ...

    def close(self) -> None: ...


class TeacherManagerState(Enum):
    """GPU residency and lifecycle state of a persistent teacher companion."""

    EMPTY = auto()
    RESIDENT = auto()
    OFFLOADED = auto()
    BROKEN = auto()
    CLOSED = auto()


class DiskCheckpointProvider:
    """Resolve teacher snapshots already available on shared storage."""

    def __init__(self, config: MOPDConfig):
        self._config = config

    def pre_fetch(self, teacher_id: str) -> None:
        self._path(teacher_id)

    def resolve(self, teacher_id: str) -> Path:
        return self._path(teacher_id)

    def consumed(self, teacher_id: str) -> None:
        del teacher_id

    def close(self) -> None:
        return

    def _path(self, teacher_id: str) -> Path:
        try:
            path = Path(self._config.teachers[teacher_id].path)
        except KeyError as exc:
            raise KeyError(f"Unknown MOPD teacher {teacher_id!r}") from exc
        if not path.is_dir():
            raise FileNotFoundError(
                f"Teacher checkpoint {teacher_id!r} is not a local directory: {path}"
            )
        return path


class PersistentTeacherManager:
    """Keep one isolated teacher controller alive across training phases."""

    def __init__(
        self,
        config: MOPDConfig,
        controller_factory: Callable[[str], TeacherController],
    ):
        self._controller_factory = controller_factory
        self._provider = DiskCheckpointProvider(config)
        self._controller: TeacherController | None = None
        self._loaded_teacher: str | None = None
        self._state = TeacherManagerState.EMPTY

    @property
    def controller(self) -> TeacherController | None:
        return self._controller

    @property
    def state(self) -> TeacherManagerState:
        return self._state

    def pre_fetch(self, teacher_id: str) -> None:
        self._ensure_usable()
        if teacher_id != self._loaded_teacher:
            self._provider.pre_fetch(teacher_id)

    def load(self, teacher_id: str) -> TeacherController:
        self._ensure_usable()
        needs_checkpoint = (
            self._state is TeacherManagerState.EMPTY
            or self._loaded_teacher != teacher_id
        )
        path = self._provider.resolve(teacher_id) if needs_checkpoint else None
        loaded = False
        try:
            if self._state is TeacherManagerState.EMPTY:
                assert path is not None
                self._controller = self._controller_factory(str(path))
                self._state = TeacherManagerState.RESIDENT
            else:
                assert self._controller is not None
                if self._state is TeacherManagerState.OFFLOADED:
                    self._controller.onload()
                    self._state = TeacherManagerState.RESIDENT
                if self._loaded_teacher != teacher_id:
                    assert path is not None
                    self._controller.load(
                        SaveLoadMeta(
                            path=str(path),
                            weight_format="hf",
                            with_optim=False,
                        )
                    )
            self._loaded_teacher = teacher_id
            loaded = True
            assert self._controller is not None
            return self._controller
        except BaseException as exc:
            self._break_controller(exc)
            raise
        finally:
            if loaded:
                self._provider.consumed(teacher_id)

    def release(self, receipt: DrainReceipt) -> None:
        self._ensure_usable()
        if not receipt.complete:
            raise RuntimeError(
                "Cannot release MOPD teacher before actor RTensor drain completes"
            )
        if self._state in (
            TeacherManagerState.EMPTY,
            TeacherManagerState.OFFLOADED,
        ):
            return
        assert self._controller is not None
        try:
            self._controller.offload()
            self._state = TeacherManagerState.OFFLOADED
        except BaseException as exc:
            self._break_controller(exc)
            raise

    def close(self) -> None:
        if self._state is TeacherManagerState.CLOSED:
            return
        try:
            self._destroy_controller()
        finally:
            try:
                self._provider.close()
            finally:
                self._state = TeacherManagerState.CLOSED

    def _break_controller(self, cause: BaseException) -> None:
        self._state = TeacherManagerState.BROKEN
        try:
            self._destroy_controller()
        except BaseException as cleanup_error:
            cause.add_note(
                "Persistent MOPD teacher cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}"
            )

    def _destroy_controller(self) -> None:
        controller = self._controller
        if controller is None:
            return
        try:
            controller.destroy()
        finally:
            self._controller = None
            self._loaded_teacher = None

    def _ensure_usable(self) -> None:
        if self._state is TeacherManagerState.CLOSED:
            raise RuntimeError("MOPD TeacherManager is closed")
        if self._state is TeacherManagerState.BROKEN:
            raise RuntimeError("Persistent MOPD teacher companion is broken")
