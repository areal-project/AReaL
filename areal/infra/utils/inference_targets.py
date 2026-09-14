# SPDX-License-Identifier: Apache-2.0

"""Persist inference metrics targets for external service discovery."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from typing import TYPE_CHECKING, Any

from areal.infra.utils.exp_metadata import get_metadata_dir
from areal.utils import logging
from areal.utils.network import format_hostport

if TYPE_CHECKING:
    from areal.api import InferenceEngine, LocalInfServerInfo

logger = logging.getLogger("InferenceTargets")

_SUPPORTED_METRICS_ENGINES = ("sglang", "vllm")


def write_inference_targets(
    *,
    inf_engine: type[InferenceEngine],
    server_infos: list[LocalInfServerInfo],
    fileroot: str | None,
    experiment_name: str | None,
    trial_name: str | None,
    role: str,
    source: str,
) -> None:
    """Write inference HTTP metrics targets in Prometheus file-SD format."""
    engine = _get_inference_metrics_engine(inf_engine)
    if engine is None:
        return

    if not fileroot or not experiment_name or not trial_name:
        logger.warning(
            "Skip writing %s targets: missing fileroot/experiment/trial.",
            engine,
        )
        return
    if not server_infos:
        logger.warning("Skip writing %s targets: server_infos is empty.", engine)
        return

    target_groups = _build_target_groups(
        engine=engine,
        server_infos=server_infos,
        role=role,
        source=source,
    )
    if not target_groups:
        logger.warning("Skip writing %s targets: no valid host/port pairs.", engine)
        return

    try:
        log_dir = get_metadata_dir(fileroot, experiment_name, trial_name)
        path = os.path.join(log_dir, f"{engine}_targets.json")
        lock_path = f"{path}.lock"
        tmp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                target_groups = _merge_inference_target_groups(
                    path=path,
                    engine=engine,
                    role=role,
                    current_groups=target_groups,
                )
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(target_groups, f, indent=2, sort_keys=True)
                    f.write("\n")
                os.replace(tmp_path, path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        logger.info(
            "%s metrics targets written to %s (%d groups, role=%s, source=%s)",
            engine,
            path,
            len(target_groups),
            role,
            source,
        )
    except Exception:
        logger.warning("Failed to write %s metrics targets.", engine, exc_info=True)


def _get_inference_metrics_engine(inf_engine: type[InferenceEngine]) -> str | None:
    engine_id = f"{inf_engine.__module__}.{inf_engine.__name__}".lower()
    for engine in _SUPPORTED_METRICS_ENGINES:
        if engine in engine_id:
            return engine
    logger.debug(
        "Skip writing inference targets for unsupported inference engine: %s",
        engine_id,
    )
    return None


def _build_target_groups(
    *,
    engine: str,
    server_infos: list[LocalInfServerInfo],
    role: str,
    source: str,
) -> list[dict[str, Any]]:
    target_groups: list[dict[str, Any]] = []
    for rank, info in enumerate(server_infos):
        if not info.host:
            continue
        target_groups.append(
            {
                "targets": [format_hostport(info.host, info.port)],
                # Target labels must not shadow inference-backend metric labels.
                "labels": {
                    "areal_backend": engine,
                    "areal_metrics_path": "/metrics",
                    "areal_rank": str(rank),
                    "areal_role": role,
                    "areal_deployment_mode": source,
                },
            }
        )
    return target_groups


def _merge_inference_target_groups(
    *,
    path: str,
    engine: str,
    role: str,
    current_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge this controller's targets with other controller roles."""
    if not os.path.exists(path):
        return current_groups

    try:
        with open(path, encoding="utf-8") as f:
            existing_groups = json.load(f)
    except Exception:
        logger.warning(
            "Failed to read existing %s targets from %s; rewriting current role.",
            engine,
            path,
            exc_info=True,
        )
        return current_groups

    if not isinstance(existing_groups, list):
        return current_groups

    merged_groups = []
    for group in existing_groups:
        if not isinstance(group, dict):
            continue
        labels = group.get("labels")
        if not isinstance(labels, dict):
            continue
        if labels.get("areal_backend") != engine:
            continue
        existing_role = labels.get("areal_role")
        if not existing_role or existing_role == role:
            continue
        merged_groups.append(group)

    return merged_groups + current_groups
