# SPDX-License-Identifier: Apache-2.0
"""Env-gated weight digests and mismatch captures for DTE debugging."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import socket
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from areal.utils import logging

logger = logging.getLogger("AwexWeightDigest")

_DIGEST_ENV = "AREAL_DTE_WEIGHT_DIGEST"
_PHASES_ENV = "AREAL_DTE_WEIGHT_DIGEST_PHASES"
_MAX_STEP_ENV = "AREAL_DTE_WEIGHT_DIGEST_MAX_STEP"
_MAX_VERSION_ENV = "AREAL_DTE_WEIGHT_DIGEST_MAX_VERSION"
_NAME_REGEX_ENV = "AREAL_DTE_WEIGHT_DIGEST_NAME_REGEX"
_MAX_TENSORS_ENV = "AREAL_DTE_WEIGHT_DIGEST_MAX_TENSORS"
_MAX_BYTES_ENV = "AREAL_DTE_WEIGHT_DIGEST_MAX_BYTES"
_SAMPLE_ELEMENTS_ENV = "AREAL_DTE_WEIGHT_DIGEST_SAMPLE_ELEMENTS"
_STRICT_ENV = "AREAL_DTE_WEIGHT_DIGEST_STRICT"
_PER_TENSOR_MANIFEST_ENV = "AREAL_DTE_WEIGHT_DIGEST_PER_TENSOR_MANIFEST"
_CAPTURE_ROOT_ENV = "AREAL_DTE_WEIGHT_CAPTURE_ROOT"
_CAPTURE_RUN_ENV = "AREAL_DTE_WEIGHT_CAPTURE_RUN"
_COMPARE_GROUP_ENV = "AREAL_DTE_WEIGHT_COMPARE_GROUP"
_ROLLING_CAPTURE_ENV = "AREAL_DTE_WEIGHT_CAPTURE_ROLLING"
_ROLLING_PHASES_ENV = "AREAL_DTE_WEIGHT_CAPTURE_ROLLING_PHASES"
_FULL_ON_MISMATCH_ENV = "AREAL_DTE_WEIGHT_CAPTURE_FULL_ON_MISMATCH"
_FAIL_FAST_ENV = "AREAL_DTE_WEIGHT_DIGEST_FAIL_FAST"
_DEFER_FAIL_PHASES_ENV = "AREAL_DTE_WEIGHT_DIGEST_DEFER_FAIL_PHASES"
_STOP_MODE_ENV = "AREAL_DTE_WEIGHT_DIGEST_STOP_MODE"
_EXPECTED_RUNS_ENV = "AREAL_DTE_WEIGHT_EXPECTED_RUNS"
_COMPARE_WAIT_SECONDS_ENV = "AREAL_DTE_WEIGHT_COMPARE_WAIT_SECONDS"
_COMPARE_POLL_SECONDS_ENV = "AREAL_DTE_WEIGHT_COMPARE_POLL_SECONDS"
_FORCE_MISMATCH_RUNS_ENV = "AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_RUNS"
_FORCE_MISMATCH_ROLE_ENV = "AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_ROLE"
_FORCE_MISMATCH_PHASE_ENV = "AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_PHASE"
_FORCE_MISMATCH_STEP_ENV = "AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_STEP"
_FORCE_MISMATCH_VERSION_ENV = "AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_VERSION"
_FORCE_MISMATCH_LABEL_ENV = "AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_LABEL"

_DEFAULT_CAPTURE_ROOT = "/storage/openpsi/users/pengzai.pyq/dte_weight_capture"
_DEFAULT_DEFER_FAIL_PHASES = "pre_send,pre_apply,post_apply"
_DEFAULT_ROLLING_PHASES = "post_optimizer_param,post_apply"
_STOP_MODE_STEP_END = "step_end"

_ROLLING_SNAPSHOTS: dict[str, dict[str, Any]] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value.strip())


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return float(value.strip())


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_") or "unknown"


def _compile_name_regex() -> re.Pattern[str] | None:
    pattern = os.environ.get(_NAME_REGEX_ENV)
    if pattern is None or pattern.strip() == "":
        return None
    return re.compile(pattern)


def weight_digest_enabled() -> bool:
    return _env_bool(_DIGEST_ENV, default=False)


def _stop_mode() -> str:
    return os.environ.get(_STOP_MODE_ENV, "").strip().lower()


def _parse_phase_filter(phases: str) -> set[str]:
    return {item.strip() for item in re.split(r"[;,]", phases) if item.strip()}


def _env_phase_set(name: str, default: str) -> set[str]:
    return _parse_phase_filter(os.environ.get(name, default))


def _expected_runs_from_env(name: str) -> set[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return set()
    return {_safe_slug(item) for item in _parse_phase_filter(value)}


def _expected_runs() -> set[str]:
    return _expected_runs_from_env(_EXPECTED_RUNS_ENV)


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value.strip())


def _phase_enabled(phase: str) -> bool:
    phases = os.environ.get(_PHASES_ENV)
    if phases is None or phases.strip() == "":
        return True
    enabled = _parse_phase_filter(phases)
    return phase in enabled


def _should_log(phase: str, *, step: int | None, version: int | None) -> bool:
    if not weight_digest_enabled() or not _phase_enabled(phase):
        return False
    max_step = _env_int(_MAX_STEP_ENV, 1)
    if step is not None and max_step >= 0 and step > max_step:
        return False
    max_version = _env_int(_MAX_VERSION_ENV, 1)
    if version is not None and max_version >= 0 and version > max_version:
        return False
    return True


def _rank_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    rank = int(os.environ.get("RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    return rank, world_size


def _git_commit() -> str:
    for name in ("AREAL_GIT_COMMIT", "EXPECTED_AREAL_COMMIT", "GIT_COMMIT"):
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(repr(value).encode("utf-8", errors="replace")).hexdigest()


def _sha256_tensor_bytes(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def _rng_context() -> dict[str, Any]:
    rng: dict[str, Any] = {
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "AREAL_DETERMINISTIC_SAMPLING": os.environ.get(
                "AREAL_DETERMINISTIC_SAMPLING", ""
            ),
            "AREAL_DETERMINISTIC_PREBUILD": os.environ.get(
                "AREAL_DETERMINISTIC_PREBUILD", ""
            ),
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": os.environ.get(
                "NVTE_ALLOW_NONDETERMINISTIC_ALGO", ""
            ),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        }
    }
    try:
        rng["python_random_state_sha256"] = _sha256_text(random.getstate())
    except Exception as exc:
        rng["python_random_state_error"] = repr(exc)
    try:
        import numpy as np

        rng["numpy_random_state_sha256"] = _sha256_text(np.random.get_state())
    except Exception as exc:
        rng["numpy_random_state_error"] = repr(exc)
    try:
        rng["torch_cpu_rng_state_sha256"] = _sha256_tensor_bytes(torch.get_rng_state())
    except Exception as exc:
        rng["torch_cpu_rng_state_error"] = repr(exc)
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            rng["torch_cuda_current_device"] = int(device)
            rng["torch_cuda_rng_state_sha256"] = _sha256_tensor_bytes(
                torch.cuda.get_rng_state(device)
            )
    except Exception as exc:
        rng["torch_cuda_rng_state_error"] = repr(exc)
    return rng


def _run_id() -> str:
    value = os.environ.get(_CAPTURE_RUN_ENV)
    if value:
        return _safe_slug(value)
    trial = os.environ.get("TRIAL_NAME")
    transfer = os.environ.get("DTE_TRANSFER")
    delta_method = os.environ.get("DTE_DELTA_METHOD")
    if trial:
        suffix = "-".join(part for part in (transfer, delta_method) if part)
        return _safe_slug(f"{trial}-{suffix}" if suffix else trial)
    return _safe_slug(f"pid{os.getpid()}-{socket.gethostname()}")


def _compare_group() -> str:
    value = os.environ.get(_COMPARE_GROUP_ENV)
    if value:
        return _safe_slug(value)
    return _safe_slug(os.environ.get("EXP_NAME", "default"))


def _capture_dir() -> Path:
    root = os.environ.get(_CAPTURE_ROOT_ENV, _DEFAULT_CAPTURE_ROOT)
    return Path(root) / _compare_group() / _run_id()


def _shared_group_dir() -> Path:
    root = os.environ.get(_CAPTURE_ROOT_ENV, _DEFAULT_CAPTURE_ROOT)
    return Path(root) / _compare_group()


def _context_record(
    *,
    role: str,
    phase: str,
    step: int | None,
    version: int | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    rank, world_size = _rank_info()
    return {
        "ts": time.time(),
        "exp": os.environ.get("EXP_NAME", ""),
        "trial": os.environ.get("TRIAL_NAME", ""),
        "job": os.environ.get("SLURM_JOB_ID", ""),
        "commit": _git_commit(),
        "capture_group": _compare_group(),
        "run": _run_id(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "phase": phase,
        "step": step,
        "version": version,
        "role": role,
        "rank": rank,
        "world_size": world_size,
        "transfer": os.environ.get("DTE_TRANSFER", ""),
        "delta_method": os.environ.get("DTE_DELTA_METHOD", ""),
        "rng": _rng_context(),
        "extra": extra or {},
    }


def _forced_mismatch_item(
    ctx: dict[str, Any],
) -> tuple[tuple[str, None], dict[str, Any]] | None:
    runs = _expected_runs_from_env(_FORCE_MISMATCH_RUNS_ENV)
    if not runs:
        return None
    if "*" not in runs and _run_id() not in runs:
        return None

    role = os.environ.get(_FORCE_MISMATCH_ROLE_ENV)
    if role and role.strip() not in {"*", str(ctx["role"])}:
        return None
    phase = os.environ.get(_FORCE_MISMATCH_PHASE_ENV)
    if phase and phase.strip() not in {"*", str(ctx["phase"])}:
        return None

    step = _env_optional_int(_FORCE_MISMATCH_STEP_ENV)
    if step is not None and ctx["step"] != step:
        return None
    version = _env_optional_int(_FORCE_MISMATCH_VERSION_ENV)
    if version is not None and ctx["version"] != version:
        return None

    label = _safe_slug(os.environ.get(_FORCE_MISMATCH_LABEL_ENV, "forced_mismatch"))
    param_name = f"__areal_dte_weight_digest_forced_mismatch__.{label}"
    meta = {
        "enabled": True,
        "label": label,
        "param": param_name,
        "runs": sorted(runs),
        "role": os.environ.get(_FORCE_MISMATCH_ROLE_ENV, ""),
        "phase": os.environ.get(_FORCE_MISMATCH_PHASE_ENV, ""),
        "step": os.environ.get(_FORCE_MISMATCH_STEP_ENV, ""),
        "version": os.environ.get(_FORCE_MISMATCH_VERSION_ENV, ""),
    }
    return (param_name, None), meta


def _tensor_payload_bytes(tensor: torch.Tensor, *, sampled: bool = True) -> bytes:
    value = tensor.detach()
    sample_elements = _env_int(_SAMPLE_ELEMENTS_ENV, -1) if sampled else -1
    if sample_elements > 0 and value.numel() > sample_elements:
        flat = value.reshape(-1)
        if sample_elements == 1:
            indices = torch.zeros(1, dtype=torch.long, device=flat.device)
        else:
            indices = (
                torch.arange(sample_elements, dtype=torch.long, device=flat.device)
                * (flat.numel() - 1)
            ) // (sample_elements - 1)
        value = flat.index_select(0, indices)
    if value.is_cuda:
        torch.cuda.synchronize(value.device)
    cpu_value = value.cpu().contiguous()
    return bytes(cpu_value.untyped_storage())


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    try:
        return int(tensor.untyped_storage().nbytes())
    except RuntimeError:
        return int(tensor.numel() * tensor.element_size())


def _digest_nbytes_estimate(tensor: torch.Tensor) -> int:
    sample_elements = _env_int(_SAMPLE_ELEMENTS_ENV, -1)
    if sample_elements > 0 and tensor.numel() > sample_elements:
        return int(sample_elements * tensor.element_size())
    return _tensor_nbytes(tensor)


def _filter_digest_items(
    items: Iterable[tuple[str, torch.Tensor | None]],
) -> tuple[list[tuple[str, torch.Tensor | None]], dict[str, Any]]:
    name_regex = _compile_name_regex()
    max_tensors = _env_int(_MAX_TENSORS_ENV, -1)
    max_bytes = _env_int(_MAX_BYTES_ENV, -1)

    selected: list[tuple[str, torch.Tensor | None]] = []
    selected_names: list[str] = []
    skipped_limit_names: list[str] = []
    skipped_name = 0
    skipped_limit = 0
    considered = 0
    selected_bytes = 0
    for name, tensor in sorted(items, key=lambda item: item[0]):
        considered += 1
        if name_regex is not None and name_regex.search(name) is None:
            skipped_name += 1
            continue
        tensor_bytes = 0 if tensor is None else _digest_nbytes_estimate(tensor)
        if max_tensors >= 0 and len(selected) >= max_tensors:
            skipped_limit += 1
            if len(skipped_limit_names) < 8:
                skipped_limit_names.append(name)
            continue
        if (
            max_bytes >= 0
            and tensor is not None
            and selected_bytes + tensor_bytes > max_bytes
        ):
            skipped_limit += 1
            if len(skipped_limit_names) < 8:
                skipped_limit_names.append(name)
            continue
        selected.append((name, tensor))
        if len(selected_names) < 8:
            selected_names.append(name)
        selected_bytes += tensor_bytes

    meta = {
        "considered": considered,
        "selected": len(selected),
        "skipped_name": skipped_name,
        "skipped_limit": skipped_limit,
        "selected_bytes_estimate": selected_bytes,
        "name_regex": "" if name_regex is None else name_regex.pattern,
        "max_tensors": max_tensors,
        "max_bytes": max_bytes,
        "sample_elements": _env_int(_SAMPLE_ELEMENTS_ENV, -1),
        "selected_names": selected_names,
        "skipped_limit_names": skipped_limit_names,
    }
    return selected, meta


def _tensor_metadata(
    name: str,
    tensor: torch.Tensor | None,
    payload: bytes | None = None,
) -> dict[str, Any]:
    if tensor is None:
        return {
            "param": name,
            "dtype": "missing",
            "shape": None,
            "numel": 0,
            "nbytes": 0,
            "digest_bytes": 0,
            "digest": "missing",
        }
    if payload is None:
        payload = _tensor_payload_bytes(tensor)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return {
        "param": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": int(tensor.numel()),
        "nbytes": _tensor_nbytes(tensor),
        "digest_bytes": len(payload),
        "digest": digest.hexdigest(),
    }


@torch.no_grad()
def build_tensor_digest_report(
    items: Iterable[tuple[str, torch.Tensor | None]],
) -> dict[str, Any]:
    """Return aggregate and per-tensor digests over named tensors."""

    digest = hashlib.sha256()
    tensor_count = 0
    missing_count = 0
    element_count = 0
    byte_count = 0
    tensor_records: list[dict[str, Any]] = []
    for name, tensor in sorted(items, key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if tensor is None:
            missing_count += 1
            tensor_records.append(_tensor_metadata(name, tensor))
            digest.update(b"<missing>")
            digest.update(b"\0")
            continue
        tensor_count += 1
        element_count += int(tensor.numel())
        payload = _tensor_payload_bytes(tensor)
        byte_count += len(payload)
        tensor_records.append(_tensor_metadata(name, tensor, payload))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return {
        "digest": digest.hexdigest(),
        "tensors": tensor_count,
        "missing": missing_count,
        "elements": element_count,
        "bytes": byte_count,
        "tensor_records": tensor_records,
    }


@torch.no_grad()
def build_tensor_digest(
    items: Iterable[tuple[str, torch.Tensor | None]],
) -> dict[str, Any]:
    """Return the legacy aggregate digest over named tensors."""

    report = build_tensor_digest_report(items)
    return {k: v for k, v in report.items() if k != "tensor_records"}


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True, default=_json_default))
            f.write("\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as f:
        json.dump(payload, f, sort_keys=True, default=_json_default)
        f.write("\n")
        tmp_name = f.name
    os.replace(tmp_name, path)


def _phase_key(
    *,
    role: str,
    phase: str,
    step: int | None,
    version: int | None,
    rank: int,
) -> str:
    step_key = "none" if step is None else str(step)
    version_key = "none" if version is None else str(version)
    return (
        f"role={_safe_slug(role)}/phase={_safe_slug(phase)}/"
        f"step={step_key}/version={version_key}/rank={rank}"
    )


def _index_path(ctx: dict[str, Any]) -> Path:
    return (
        _shared_group_dir()
        / "index"
        / _phase_key(
            role=str(ctx["role"]),
            phase=str(ctx["phase"]),
            step=ctx["step"],
            version=ctx["version"],
            rank=int(ctx["rank"]),
        )
        / f"{_run_id()}.json"
    )


def _load_peer_reports(path: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for peer in sorted(path.parent.glob("*.json")):
        if peer.name == path.name:
            continue
        try:
            with peer.open("r", encoding="utf-8") as f:
                reports.append(json.load(f))
        except Exception:
            logger.warning(
                "Failed to load peer digest report from %s",
                peer,
                exc_info=True,
            )
    return reports


def _wait_for_expected_peer_reports(
    path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    expected = _expected_runs()
    if not expected:
        return _load_peer_reports(path), []

    current_run = _run_id()
    if current_run not in expected:
        expected.add(current_run)
    required = sorted(expected - {current_run})
    wait_seconds = max(0.0, _env_float(_COMPARE_WAIT_SECONDS_ENV, 0.0))
    poll_seconds = max(0.05, _env_float(_COMPARE_POLL_SECONDS_ENV, 1.0))
    deadline = time.monotonic() + wait_seconds

    missing: list[str] = required
    while True:
        missing = [
            run for run in required if not (path.parent / f"{run}.json").exists()
        ]
        if not missing:
            break
        if _existing_mismatch_flag() is not None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_seconds, remaining))

    reports: list[dict[str, Any]] = []
    for run in required:
        peer = path.parent / f"{run}.json"
        if not peer.exists():
            continue
        try:
            with peer.open("r", encoding="utf-8") as f:
                reports.append(json.load(f))
        except Exception:
            logger.warning(
                "Failed to load expected peer digest %s",
                peer,
                exc_info=True,
            )
            missing.append(run)
    return reports, sorted(set(missing))


def _find_mismatch(
    current: dict[str, Any],
    peers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_report = current["report"]
    current_tensors = {
        item["param"]: item for item in current_report.get("tensor_records", [])
    }
    mismatches: list[dict[str, Any]] = []
    peer_summaries: list[dict[str, Any]] = []
    for peer in peers:
        peer_report = peer.get("report", {})
        peer_summaries.append(
            {
                "run": peer.get("context", {}).get("run", ""),
                "digest": peer_report.get("digest", ""),
                "path": peer.get("index_path", ""),
            }
        )
        if peer_report.get("digest") == current_report.get("digest"):
            continue
        peer_tensors = {
            item["param"]: item for item in peer_report.get("tensor_records", [])
        }
        names = sorted(set(current_tensors) | set(peer_tensors))
        for name in names:
            left = current_tensors.get(name)
            right = peer_tensors.get(name)
            if (
                left is None
                or right is None
                or left.get("digest") != right.get("digest")
            ):
                mismatches.append(
                    {
                        "param": name,
                        "current_digest": None if left is None else left.get("digest"),
                        "peer_digest": None if right is None else right.get("digest"),
                        "peer_run": peer.get("context", {}).get("run", ""),
                    }
                )
                if len(mismatches) >= 128:
                    break
        if mismatches:
            break
    if not mismatches:
        return None
    return {
        "reason": "peer_digest_mismatch",
        "mismatch_count_sample": len(mismatches),
        "mismatches": mismatches,
        "peers": peer_summaries,
    }


def _existing_mismatch_flag() -> dict[str, Any] | None:
    flag_path = _shared_group_dir() / "mismatch_flag.json"
    if not flag_path.exists():
        return None
    try:
        with flag_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Failed to read mismatch flag from %s", flag_path, exc_info=True)
        return {"reason": "mismatch_flag_unreadable", "path": str(flag_path)}


def _rollout_artifact_paths() -> dict[str, dict[str, Any]]:
    artifacts = {}
    for name in (
        "rollout_fingerprint.jsonl",
        "rollout_seed_manifest.jsonl",
        "rollout_batch_manifest.jsonl",
        "train_step_manifest.jsonl",
    ):
        path = _capture_dir() / name
        artifacts[name] = {"path": str(path), "exists_at_summary": path.exists()}
    return artifacts


def step_end_stop_enabled() -> bool:
    return weight_digest_enabled() and _stop_mode() == _STOP_MODE_STEP_END


def raise_if_step_end_mismatch(step: int | None = None) -> None:
    if not step_end_stop_enabled():
        return
    flag = _existing_mismatch_flag()
    if flag is None:
        return
    flag_path = _shared_group_dir() / "mismatch_flag.json"
    logger.error(
        "DTE_WEIGHT_DIGEST step_end stop requested after step=%s flag=%s",
        "none" if step is None else step,
        flag_path,
    )
    raise RuntimeError(
        "DTE weight digest mismatch detected at step end; "
        f"flag={flag_path} step={step} capture_dir={_capture_dir()}"
    )


def _clone_snapshot_items(
    items: Iterable[tuple[str, torch.Tensor | None]],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in sorted(items, key=lambda item: item[0]):
        if tensor is None:
            continue
        value = tensor.detach()
        if value.is_cuda:
            torch.cuda.synchronize(value.device)
        tensors[name] = value.cpu().clone()
    return tensors


def _snapshot_key(ctx: dict[str, Any]) -> str:
    return f"{ctx['role']}:{ctx['phase']}:{ctx['rank']}"


def _update_rolling_snapshot(
    ctx: dict[str, Any],
    items: list[tuple[str, torch.Tensor | None]],
    report: dict[str, Any],
) -> None:
    if not _env_bool(_ROLLING_CAPTURE_ENV, default=False):
        return
    if ctx["phase"] not in _env_phase_set(_ROLLING_PHASES_ENV, _DEFAULT_ROLLING_PHASES):
        return
    try:
        _ROLLING_SNAPSHOTS[_snapshot_key(ctx)] = {
            "context": dict(ctx),
            "report": {k: v for k, v in report.items() if k != "tensor_records"},
            "tensor_records": report.get("tensor_records", []),
            "tensors": _clone_snapshot_items(items),
        }
    except Exception:
        logger.exception(
            "DTE_WEIGHT_DIGEST rolling snapshot failed role=%s phase=%s",
            ctx["role"],
            ctx["phase"],
        )
        if _env_bool(_STRICT_ENV, default=False):
            raise


def _dump_snapshot(
    ctx: dict[str, Any],
    *,
    label: str,
    tensors: dict[str, torch.Tensor],
    report: dict[str, Any],
    mismatch: dict[str, Any],
) -> Path:
    rank = int(ctx["rank"])
    step_key = "none" if ctx["step"] is None else str(ctx["step"])
    version_key = "none" if ctx["version"] is None else str(ctx["version"])
    out_dir = (
        _capture_dir()
        / "captures"
        / f"step={step_key}"
        / f"version={version_key}"
        / _safe_slug(str(ctx["role"]))
        / _safe_slug(str(ctx["phase"]))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}-rank{rank}.pt"
    torch.save(
        {
            "context": ctx,
            "report": report,
            "mismatch": mismatch,
            "tensors": tensors,
        },
        path,
    )
    return path


def _dump_mismatch_capture(
    ctx: dict[str, Any],
    items: list[tuple[str, torch.Tensor | None]],
    report: dict[str, Any],
    mismatch: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_path = None
    if _env_bool(_FULL_ON_MISMATCH_ENV, default=True):
        current_tensors = _clone_snapshot_items(items)
        current_path = _dump_snapshot(
            ctx,
            label="current",
            tensors=current_tensors,
            report=report,
            mismatch=mismatch,
        )
        records.append(
            {
                "record_type": "capture",
                **ctx,
                "label": "current",
                "path": str(current_path),
                "params": sorted(current_tensors),
                "mismatch": mismatch,
            }
        )

    previous_paths: list[str] = []
    for key, snapshot in sorted(_ROLLING_SNAPSHOTS.items()):
        previous_path = _dump_snapshot(
            snapshot["context"],
            label="previous",
            tensors=snapshot["tensors"],
            report=snapshot["report"],
            mismatch=mismatch,
        )
        previous_paths.append(str(previous_path))
        records.append(
            {
                "record_type": "capture",
                **snapshot["context"],
                "label": "previous",
                "path": str(previous_path),
                "params": sorted(snapshot["tensors"]),
                "mismatch": mismatch,
                "snapshot_key": key,
            }
        )

    summary = {
        "context": ctx,
        "mismatch": mismatch,
        "current_path": "" if current_path is None else str(current_path),
        "previous_paths": previous_paths,
        "rollout_artifacts": _rollout_artifact_paths(),
    }
    summary_path = _capture_dir() / "mismatch_summary.json"
    _write_json_atomic(summary_path, summary)
    flag_path = _shared_group_dir() / "mismatch_flag.json"
    _write_json_atomic(flag_path, summary)
    records.append(
        {
            "record_type": "mismatch_summary",
            **ctx,
            "path": str(summary_path),
            "flag_path": str(flag_path),
            "mismatch": mismatch,
        }
    )
    return records


def _local_or_global_mismatch(local_mismatch: bool) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return local_mismatch
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    flag = torch.tensor([1 if local_mismatch else 0], dtype=torch.int32, device=device)
    try:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    except Exception:
        logger.warning("DTE_WEIGHT_DIGEST mismatch all_reduce failed", exc_info=True)
        if _env_bool(_STRICT_ENV, default=False):
            raise
        return local_mismatch
    return bool(flag.item())


def _should_fail_fast(phase: str) -> bool:
    if _stop_mode() == _STOP_MODE_STEP_END:
        return False
    if not _env_bool(_FAIL_FAST_ENV, default=True):
        return False
    return phase not in _env_phase_set(
        _DEFER_FAIL_PHASES_ENV,
        _DEFAULT_DEFER_FAIL_PHASES,
    )


def log_tensor_digest(
    items: Iterable[tuple[str, torch.Tensor | None]],
    *,
    role: str,
    phase: str,
    step: int | None = None,
    version: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log rank-local tensor digests and capture mismatch state when enabled."""

    if not _should_log(phase, step=step, version=version):
        return
    try:
        filtered_items, filter_meta = _filter_digest_items(items)
        filtered_items = list(filtered_items)
        rank, world_size = _rank_info()
        ctx = _context_record(
            role=role,
            phase=phase,
            step=step,
            version=version,
            extra={**(extra or {}), "filter": filter_meta},
        )
        forced_item = _forced_mismatch_item(ctx)
        if forced_item is not None:
            item, forced_meta = forced_item
            filtered_items.append(item)
            ctx["extra"] = {**ctx["extra"], "forced_mismatch": forced_meta}
        logger.info(
            "DTE_WEIGHT_DIGEST_BEGIN role=%s phase=%s step=%s version=%s "
            "rank=%s world_size=%s filter=%s extra=%s",
            role,
            phase,
            "none" if step is None else step,
            "none" if version is None else version,
            rank,
            world_size,
            filter_meta,
            extra or {},
        )

        report = build_tensor_digest_report(filtered_items)
        aggregate_record = {
            "record_type": "aggregate",
            **ctx,
            "digest": report["digest"],
            "tensors": report["tensors"],
            "missing": report["missing"],
            "elements": report["elements"],
            "bytes": report["bytes"],
        }
        manifest_path = _capture_dir() / "manifest.jsonl"
        manifest_records = [aggregate_record]
        if _env_bool(_PER_TENSOR_MANIFEST_ENV, default=True):
            manifest_records.extend(
                {
                    "record_type": "tensor",
                    **ctx,
                    "param": tensor_record["param"],
                    "dtype": tensor_record["dtype"],
                    "shape": tensor_record["shape"],
                    "numel": tensor_record["numel"],
                    "nbytes": tensor_record["nbytes"],
                    "digest_bytes": tensor_record["digest_bytes"],
                    "digest": tensor_record["digest"],
                    "path": "",
                }
                for tensor_record in report.get("tensor_records", [])
            )

        index_path = _index_path(ctx)
        current_index = {
            "context": ctx,
            "filter": filter_meta,
            "report": report,
            "index_path": str(index_path),
        }
        _write_json_atomic(index_path, current_index)
        peers, missing_peers = _wait_for_expected_peer_reports(index_path)
        if missing_peers:
            mismatch = {
                "reason": "expected_peer_timeout",
                "missing_runs": missing_peers,
                "expected_runs": sorted(_expected_runs()),
                "wait_seconds": _env_float(_COMPARE_WAIT_SECONDS_ENV, 0.0),
                "mismatches": [],
                "peers": [
                    {
                        "run": peer.get("context", {}).get("run", ""),
                        "digest": peer.get("report", {}).get("digest", ""),
                        "path": peer.get("index_path", ""),
                    }
                    for peer in peers
                ],
            }
        else:
            mismatch = _find_mismatch(current_index, peers)
        existing_flag = _existing_mismatch_flag()
        if mismatch is None and existing_flag is not None:
            mismatch = {
                "reason": "mismatch_flag_observed",
                "flag": existing_flag,
                "mismatches": [],
                "peers": [],
            }

        global_mismatch = _local_or_global_mismatch(mismatch is not None)
        if global_mismatch:
            if mismatch is None:
                mismatch = {
                    "reason": "rank_peer_reported_mismatch",
                    "mismatches": [],
                    "peers": [],
                }
            manifest_records.extend(
                _dump_mismatch_capture(ctx, filtered_items, report, mismatch)
            )
        _append_jsonl(manifest_path, manifest_records)
        if not global_mismatch:
            _update_rolling_snapshot(ctx, filtered_items, report)

        logger.info(
            "DTE_WEIGHT_DIGEST role=%s phase=%s step=%s version=%s "
            "rank=%s world_size=%s tensors=%d missing=%d elements=%d "
            "bytes=%d digest=%s manifest=%s mismatch=%s filter=%s extra=%s",
            role,
            phase,
            "none" if step is None else step,
            "none" if version is None else version,
            rank,
            world_size,
            report["tensors"],
            report["missing"],
            report["elements"],
            report["bytes"],
            report["digest"],
            manifest_path,
            global_mismatch,
            filter_meta,
            extra or {},
        )
        if global_mismatch and _should_fail_fast(phase):
            raise RuntimeError(
                "DTE weight digest mismatch detected; "
                f"capture_dir={_capture_dir()} phase={phase} "
                f"step={step} version={version} role={role} rank={rank}"
            )
    except Exception:
        logger.exception(
            "DTE_WEIGHT_DIGEST failed role=%s phase=%s step=%s version=%s",
            role,
            phase,
            step,
            version,
        )
        if _env_bool(_STRICT_ENV, default=False):
            raise
