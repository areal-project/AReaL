# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum, auto
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from areal.api import SaveLoadMeta
from areal.api.cli_args import MOPDConfig
from areal.infra.rpc.rtensor import RTensorDrainReceipt


class TeacherController(Protocol):
    def compute_logp_padded(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[Any] | None, list[Any]]: ...

    def assert_mopd_runtime_topology(self) -> None: ...

    def load(self, meta: SaveLoadMeta) -> None: ...

    def onload(self) -> None: ...

    def offload(self) -> None: ...

    def strict_clear_batches(self, *targets: Any) -> RTensorDrainReceipt: ...

    def destroy(self) -> None: ...


class TeacherManager(Protocol):
    def pre_fetch(self, teacher_id: str) -> None: ...

    def load(self, teacher_id: str) -> TeacherController: ...

    def release(self, receipt: RTensorDrainReceipt) -> None: ...

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


class LocalMemoryCheckpointProvider:
    """Stage at most one next teacher snapshot using atomic ready directories."""

    _RUN_PREFIX = ".run-"

    def __init__(self, config: MOPDConfig):
        self._config = config
        self._root = Path(config.manager.staging_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._sweep_stale_runs()
        self._run_dir = (
            self._root / f"{self._RUN_PREFIX}{os.getpid()}-{uuid.uuid4().hex}"
        )
        self._run_dir.mkdir()
        self._write_manifest()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mopd-copy"
        )
        self._future: Future[Path] | None = None
        self._future_teacher: str | None = None
        self._ready_paths: dict[str, Path] = {}
        self._closed = False
        self._lock = Lock()

    def pre_fetch(self, teacher_id: str) -> None:
        source = self._source_path(teacher_id)
        with self._lock:
            self._ensure_open()
            if teacher_id in self._ready_paths or self._future_teacher == teacher_id:
                return
            if self._ready_paths:
                ready_teacher = next(iter(self._ready_paths))
                raise RuntimeError(
                    "Local-memory MOPD provider already holds ready checkpoint "
                    f"{ready_teacher!r}"
                )
            if self._future is not None:
                if not self._future.done():
                    raise RuntimeError(
                        "Local-memory MOPD provider already has one checkpoint in flight"
                    )
                self._finalize_future_locked()
                if self._ready_paths:
                    raise RuntimeError(
                        "Local-memory MOPD provider already holds one ready checkpoint"
                    )
            self._check_capacity(source)
            self._future_teacher = teacher_id
            self._future = self._executor.submit(
                self._copy_snapshot, teacher_id, source
            )

    def resolve(self, teacher_id: str) -> Path:
        with self._lock:
            self._ensure_open()
            needs_prefetch = (
                teacher_id not in self._ready_paths
                and self._future_teacher != teacher_id
            )
        if needs_prefetch:
            self.pre_fetch(teacher_id)
        with self._lock:
            self._finalize_future_locked(expected_teacher=teacher_id)
            try:
                return self._ready_paths[teacher_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"Teacher checkpoint {teacher_id!r} was not staged"
                ) from exc

    def consumed(self, teacher_id: str) -> None:
        with self._lock:
            path = self._ready_paths.pop(teacher_id, None)
        if path is not None:
            shutil.rmtree(path, ignore_errors=False)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future = self._future
        if future is not None:
            future.cancel()
            if not future.cancelled():
                try:
                    future.result()
                except Exception:  # noqa: BLE001
                    pass
        self._executor.shutdown(wait=True, cancel_futures=True)
        shutil.rmtree(self._run_dir, ignore_errors=True)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Local-memory MOPD provider is closed")

    def _source_path(self, teacher_id: str) -> Path:
        try:
            path = Path(self._config.teachers[teacher_id].path)
        except KeyError as exc:
            raise KeyError(f"Unknown MOPD teacher {teacher_id!r}") from exc
        if not path.is_dir():
            raise FileNotFoundError(
                f"Teacher checkpoint {teacher_id!r} is not a local directory: {path}"
            )
        return path

    def _check_capacity(self, source: Path) -> None:
        required = sum(
            path.stat().st_size for path in source.rglob("*") if path.is_file()
        )
        reserve = self._config.manager.min_free_bytes or 0
        available = shutil.disk_usage(self._root).free
        if available < required + reserve:
            raise OSError(
                f"Insufficient staging space under {self._root}: need "
                f"{required + reserve} bytes, have {available}"
            )

    def _copy_snapshot(self, teacher_id: str, source: Path) -> Path:
        tmp = self._run_dir / f"{teacher_id}.tmp.{uuid.uuid4().hex}"
        ready = self._run_dir / f"{teacher_id}.ready"
        try:
            shutil.copytree(source, tmp)
            self._fsync_tree(tmp)
            os.replace(tmp, ready)
            self._fsync_directory(self._run_dir)
            return ready
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def _finalize_future_locked(self, expected_teacher: str | None = None) -> None:
        if self._future is None:
            return
        teacher_id = self._future_teacher
        if expected_teacher is not None and teacher_id != expected_teacher:
            raise RuntimeError(
                f"Checkpoint {teacher_id!r} is staged, not {expected_teacher!r}"
            )
        ready = self._future.result()
        assert teacher_id is not None
        self._ready_paths[teacher_id] = ready
        self._future = None
        self._future_teacher = None

    def _write_manifest(self) -> None:
        manifest = self._run_dir / "owner.json"
        manifest.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        with manifest.open("rb") as stream:
            os.fsync(stream.fileno())
        self._fsync_directory(self._run_dir)

    def _sweep_stale_runs(self) -> None:
        for run_dir in self._root.glob(f"{self._RUN_PREFIX}*"):
            manifest = run_dir / "owner.json"
            try:
                owner_pid = int(json.loads(manifest.read_text(encoding="utf-8"))["pid"])
            except (
                FileNotFoundError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                continue
            if not _pid_exists(owner_pid):
                shutil.rmtree(run_dir, ignore_errors=True)

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        for path in root.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        for path in sorted(
            (entry for entry in root.rglob("*") if entry.is_dir()),
            key=lambda entry: len(entry.parts),
            reverse=True,
        ):
            LocalMemoryCheckpointProvider._fsync_directory(path)
        LocalMemoryCheckpointProvider._fsync_directory(root)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PersistentTeacherManager:
    """Keep one isolated teacher controller alive across training phases."""

    def __init__(
        self,
        config: MOPDConfig,
        controller_factory: Callable[[str], TeacherController],
    ):
        self._controller_factory = controller_factory
        self._provider = (
            DiskCheckpointProvider(config)
            if config.manager.type == "disk"
            else LocalMemoryCheckpointProvider(config)
        )
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

    def release(self, receipt: RTensorDrainReceipt) -> None:
        self._ensure_usable()
        if receipt.consumer_role != "actor":
            raise RuntimeError(
                "Cannot release MOPD teacher without an actor RTensor drain receipt"
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
