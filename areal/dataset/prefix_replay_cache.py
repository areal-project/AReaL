# SPDX-License-Identifier: Apache-2.0

"""Processed-cache helpers for prefix replay datasets."""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from areal.utils import logging

logger = logging.getLogger("PrefixReplayCache")

_CACHE_VERSION = 4
_CACHE_META_FILE = ".meta.json"
_CACHE_DONE_MARKER = ".done"
_CACHE_TRAJECTORIES_DIR = "trajectories"
_CACHE_INDICES_DIR = "indices"
_CACHE_ATTEMPTS_DIR = "attempts"
_CACHE_GENERATION_FILE = ".generation"
_CACHE_REQUEST_FILE = "request.json"


class ProcessedPrefixReplayCacheLock:
    """Exclusive lease for one processed prefix replay cache directory."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.lock_path = self.cache_dir.with_name(f"{self.cache_dir.name}.lock")
        self._lock_file = None

    def acquire(self) -> ProcessedPrefixReplayCacheLock:
        import fcntl

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_file.close()
            self._lock_file = None
            raise RuntimeError(
                f"Processed prefix replay cache is already in use: {self.cache_dir}. "
                "Use a different trial_name/cache_dir or wait for the active job."
            ) from exc

        self._lock_file.seek(0)
        self._lock_file.truncate()
        json.dump(
            {
                "cache_dir": str(self.cache_dir),
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "time": time.time(),
            },
            self._lock_file,
        )
        self._lock_file.write("\n")
        self._lock_file.flush()
        os.fsync(self._lock_file.fileno())
        logger.info("Acquired processed prefix replay cache lock: %s", self.lock_path)
        return self

    def close(self) -> None:
        if self._lock_file is None:
            return

        import fcntl

        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lock_file.close()
            self._lock_file = None
        logger.info("Released processed prefix replay cache lock: %s", self.lock_path)


def jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def source_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    fingerprint: dict[str, Any] = {"path": str(source)}
    try:
        stat = source.stat()
    except OSError:
        return fingerprint
    fingerprint.update(
        {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "is_dir": source.is_dir(),
        }
    )
    return fingerprint


def build_prefix_replay_cache_metadata(
    path: str,
    *,
    split: str,
    tokenizer_path: str,
    max_length: int,
    kappa: float,
    seed: int,
    input_mode: str,
    drop_system_messages: bool,
    route_field: str | None,
    route_metadata_field: str | None,
    route_default_value: str | None,
    parse_tool_call_args: bool,
    chat_template_kwargs: dict[str, Any],
    experience_path: str | None = None,
    experience_field: str | None = None,
    experience_context_length: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "version": _CACHE_VERSION,
        "split": split,
        "source": source_fingerprint(path),
        "tokenizer_path": tokenizer_path,
        "max_length": max_length,
        "kappa": kappa,
        "seed": seed,
        "input_mode": input_mode,
        "drop_system_messages": drop_system_messages,
        "route_field": route_field,
        "route_metadata_field": route_metadata_field,
        "route_default_value": route_default_value,
        "parse_tool_call_args": parse_tool_call_args,
        "chat_template_kwargs": chat_template_kwargs,
    }
    if experience_path is not None:
        metadata["experience"] = {
            "source": source_fingerprint(experience_path),
            "field": experience_field,
            "context_length": experience_context_length,
        }
    return jsonable(metadata)


def validate_prefix_replay_cache(
    cache_dir: str | Path,
    expected_meta: dict[str, Any],
) -> tuple[bool, str]:
    cache_path = Path(cache_dir)
    done_path = cache_path / _CACHE_DONE_MARKER
    meta_path = cache_path / _CACHE_META_FILE
    trajectories_path = cache_path / _CACHE_TRAJECTORIES_DIR
    indices_path = cache_path / _CACHE_INDICES_DIR
    if not done_path.is_file():
        return False, "completion marker is missing"
    if not meta_path.is_file():
        return False, "cache metadata is missing"
    if not trajectories_path.is_dir():
        return False, "cache trajectories dataset directory is missing"
    if not indices_path.is_dir():
        return False, "cache indices dataset directory is missing"

    try:
        with meta_path.open(encoding="utf-8") as meta_file:
            cached_meta = json.load(meta_file)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cache metadata is unreadable: {exc}"
    if cached_meta != expected_meta:
        return False, "cache metadata does not match current prefix replay settings"

    try:
        with done_path.open(encoding="utf-8") as done_file:
            done_meta = json.load(done_file)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"completion metadata is unreadable: {exc}"
    if done_meta.get("version") != _CACHE_VERSION:
        return False, "cache version is incompatible"
    if not isinstance(done_meta.get("rows"), int) or done_meta["rows"] <= 0:
        return False, "cached row count is invalid"
    if (
        not isinstance(done_meta.get("trajectories"), int)
        or done_meta["trajectories"] <= 0
    ):
        return False, "cached trajectory count is invalid"

    try:
        from datasets import load_from_disk

        trajectories = load_from_disk(str(trajectories_path))
        indices = load_from_disk(str(indices_path))
        if "row_json" not in trajectories.column_names:
            return False, "cached trajectories dataset is missing row_json"
        if "row_json" not in indices.column_names:
            return False, "cached indices dataset is missing row_json"
        if len(trajectories) != done_meta["trajectories"]:
            return (
                False,
                "cached trajectory count does not match completion marker",
            )
        if len(indices) != done_meta["rows"]:
            return False, "cached index row count does not match completion marker"
    except Exception as exc:
        return False, f"cached datasets are unreadable: {exc}"

    return True, "complete cache is compatible"


def preflight_prefix_replay_cache(
    cache_dir: str | Path,
    expected_meta: dict[str, Any],
) -> tuple[bool, str]:
    cache_path = Path(cache_dir)
    cache_valid, reason = validate_prefix_replay_cache(cache_path, expected_meta)
    if cache_valid:
        logger.info("Preflight: reusing compatible prefix replay cache: %s", cache_path)
        return True, reason

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() or cache_path.is_symlink():
        stale_path = cache_path.with_name(
            f"{cache_path.name}.stale.{int(time.time())}.{os.getpid()}"
        )
        try:
            os.replace(cache_path, stale_path)
        except FileNotFoundError:
            pass
        else:
            if stale_path.is_symlink() or stale_path.is_file():
                stale_path.unlink(missing_ok=True)
            else:
                shutil.rmtree(stale_path, ignore_errors=True)
            logger.info(
                "Preflight: removed stale prefix replay cache %s: %s",
                cache_path,
                reason,
            )

    cache_path.mkdir(parents=True, exist_ok=True)
    return False, reason


def prepare_prefix_replay_cache_attempt(
    cache_dir: str | Path,
    request: dict[str, Any],
) -> str:
    """Publish a generation-scoped request for distributed preprocessing."""

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    generation = uuid.uuid4().hex
    attempt_path = cache_path / _CACHE_ATTEMPTS_DIR / generation
    attempt_path.mkdir(parents=True, exist_ok=False)

    request_path = attempt_path / _CACHE_REQUEST_FILE
    request_tmp = request_path.with_name(f"{request_path.name}.tmp.{os.getpid()}")
    generation_path = cache_path / _CACHE_GENERATION_FILE
    generation_tmp = generation_path.with_name(
        f"{generation_path.name}.tmp.{os.getpid()}"
    )
    try:
        with request_tmp.open("w", encoding="utf-8") as request_file:
            json.dump(
                jsonable(request), request_file, sort_keys=True, ensure_ascii=False
            )
            request_file.flush()
            os.fsync(request_file.fileno())
        os.replace(request_tmp, request_path)

        with generation_tmp.open("w", encoding="utf-8") as generation_file:
            generation_file.write(f"{generation}\n")
            generation_file.flush()
            os.fsync(generation_file.fileno())
        os.replace(generation_tmp, generation_path)
    finally:
        request_tmp.unlink(missing_ok=True)
        generation_tmp.unlink(missing_ok=True)

    logger.info(
        "Prepared distributed prefix replay cache generation %s: %s",
        generation,
        cache_path,
    )
    return generation


def get_prefix_replay_cache_generation(cache_dir: str | Path) -> str:
    """Return the current controller-published cache generation."""

    generation_path = Path(cache_dir) / _CACHE_GENERATION_FILE
    try:
        generation = generation_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(
            f"Prefix replay cache generation is missing: {cache_dir}"
        ) from exc
    if len(generation) != 32 or any(
        character not in "0123456789abcdef" for character in generation
    ):
        raise RuntimeError(f"Prefix replay cache generation is invalid: {cache_dir}")
    return generation


def prefix_replay_cache_attempt_dir(
    cache_dir: str | Path,
    generation: str,
) -> Path:
    """Return one generation's distributed preprocessing artifact directory."""

    if (
        len(generation) != 32
        or any(character not in "0123456789abcdef" for character in generation)
        or Path(generation).name != generation
    ):
        raise ValueError(f"Invalid prefix replay cache generation: {generation!r}")
    return Path(cache_dir) / _CACHE_ATTEMPTS_DIR / generation


def read_prefix_replay_cache_request(
    cache_dir: str | Path,
    generation: str,
) -> dict[str, Any]:
    """Read the immutable request associated with a cache generation."""

    current_generation = get_prefix_replay_cache_generation(cache_dir)
    if current_generation != generation:
        raise RuntimeError(
            f"Prefix replay cache generation {generation} was superseded by "
            f"{current_generation}"
        )
    request_path = (
        prefix_replay_cache_attempt_dir(cache_dir, generation) / _CACHE_REQUEST_FILE
    )
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Distributed prefix replay request is unreadable: {request_path}"
        ) from exc
    if not isinstance(request, dict):
        raise TypeError(
            f"Distributed prefix replay request must be a JSON object: {request_path}"
        )
    return request


def read_prefix_replay_cache(
    cache_dir: str | Path,
    expected_meta: dict[str, Any],
):
    cache_path = Path(cache_dir)
    cache_valid, reason = validate_prefix_replay_cache(cache_path, expected_meta)
    if not cache_valid:
        raise ValueError(f"prefix replay cache is not valid: {reason}")

    from datasets import load_from_disk

    from areal.dataset.prefix_replay import PrefixReplayIndexedDataset

    trajectory_rows = _rows_from_json_dataset(
        load_from_disk(str(cache_path / _CACHE_TRAJECTORIES_DIR))
    )
    index_rows = _rows_from_json_dataset(
        load_from_disk(str(cache_path / _CACHE_INDICES_DIR))
    )
    if not trajectory_rows or not index_rows:
        raise ValueError(f"cached prefix replay dataset is empty: {cache_path}")
    dataset = PrefixReplayIndexedDataset(trajectory_rows, index_rows)
    logger.info("Loaded %d cached replay prefixes from %s", len(dataset), cache_path)
    return dataset


def _json_dataset_from_rows(rows: list[dict[str, Any]]):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "row_json": [
                json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows
            ]
        }
    )


def _rows_from_json_dataset(dataset) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in dataset["row_json"]:
        row = json.loads(value)
        if not isinstance(row, dict):
            raise TypeError("cached prefix replay row is not a JSON object")
        rows.append(row)
    return rows


def write_prefix_replay_cache(
    cache_dir: str | Path,
    meta: dict[str, Any],
    rows,
    *,
    expected_generation: str | None = None,
) -> None:
    from areal.dataset.prefix_replay import PrefixReplayIndexedDataset

    if not isinstance(rows, PrefixReplayIndexedDataset):
        if not rows:
            raise ValueError("refusing to cache an empty prefix replay dataset")
        rows = PrefixReplayIndexedDataset.from_prefix_rows(rows)
    if len(rows) == 0:
        raise ValueError("refusing to cache an empty prefix replay dataset")
    trajectory_rows, index_rows = rows.to_cache_rows()

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    def _assert_expected_generation() -> None:
        if expected_generation is None:
            return
        current_generation = get_prefix_replay_cache_generation(cache_path)
        if current_generation != expected_generation:
            raise RuntimeError(
                f"Prefix replay cache generation {expected_generation} was "
                f"superseded by {current_generation}"
            )

    trajectories_path = cache_path / _CACHE_TRAJECTORIES_DIR
    indices_path = cache_path / _CACHE_INDICES_DIR
    trajectories_tmp = cache_path / f"{_CACHE_TRAJECTORIES_DIR}.tmp.{os.getpid()}"
    indices_tmp = cache_path / f"{_CACHE_INDICES_DIR}.tmp.{os.getpid()}"
    meta_tmp = cache_path / f"{_CACHE_META_FILE}.tmp.{os.getpid()}"
    done_tmp = cache_path / f"{_CACHE_DONE_MARKER}.tmp.{os.getpid()}"
    try:
        shutil.rmtree(trajectories_tmp, ignore_errors=True)
        shutil.rmtree(indices_tmp, ignore_errors=True)
        _json_dataset_from_rows(trajectory_rows).save_to_disk(str(trajectories_tmp))
        _json_dataset_from_rows(index_rows).save_to_disk(str(indices_tmp))
        with meta_tmp.open("w", encoding="utf-8") as meta_file:
            json.dump(meta, meta_file, sort_keys=True, ensure_ascii=False)
            meta_file.flush()
            os.fsync(meta_file.fileno())
        with done_tmp.open("w", encoding="utf-8") as done_file:
            json.dump(
                {
                    "version": _CACHE_VERSION,
                    "rows": len(index_rows),
                    "trajectories": len(trajectory_rows),
                },
                done_file,
            )
            done_file.flush()
            os.fsync(done_file.fileno())

        _assert_expected_generation()
        if trajectories_path.exists():
            stale_trajectories_path = (
                cache_path / f"{_CACHE_TRAJECTORIES_DIR}.stale.{os.getpid()}"
            )
            os.replace(trajectories_path, stale_trajectories_path)
            shutil.rmtree(stale_trajectories_path, ignore_errors=True)
        if indices_path.exists():
            stale_indices_path = (
                cache_path / f"{_CACHE_INDICES_DIR}.stale.{os.getpid()}"
            )
            os.replace(indices_path, stale_indices_path)
            shutil.rmtree(stale_indices_path, ignore_errors=True)
        os.replace(trajectories_tmp, trajectories_path)
        os.replace(indices_tmp, indices_path)
        os.replace(meta_tmp, cache_path / _CACHE_META_FILE)
        _assert_expected_generation()
        os.replace(done_tmp, cache_path / _CACHE_DONE_MARKER)
    finally:
        shutil.rmtree(trajectories_tmp, ignore_errors=True)
        shutil.rmtree(indices_tmp, ignore_errors=True)
        meta_tmp.unlink(missing_ok=True)
        done_tmp.unlink(missing_ok=True)

    logger.info(
        "Saved %d processed replay prefixes from %d trajectories to %s",
        len(index_rows),
        len(trajectory_rows),
        cache_path,
    )


__all__ = [
    "ProcessedPrefixReplayCacheLock",
    "build_prefix_replay_cache_metadata",
    "get_prefix_replay_cache_generation",
    "prefix_replay_cache_attempt_dir",
    "preflight_prefix_replay_cache",
    "prepare_prefix_replay_cache_attempt",
    "read_prefix_replay_cache",
    "read_prefix_replay_cache_request",
    "validate_prefix_replay_cache",
    "write_prefix_replay_cache",
]
