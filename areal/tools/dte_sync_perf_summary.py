# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

GATEWAY_RE = re.compile(
    r"Weight update completed for pair 'actor-rollout' v(?P<version>\d+) "
    r"\((?P<duration_ms>[0-9.]+)ms\)"
)
TRAINER_UPDATE_RE = re.compile(
    r"timeperf/update_weights\s*(?:\||│|=|:)\s*(?P<value>[+-]?[0-9.eE+-]+)"
)
NUMERIC_RE = re.compile(r"^[+-]?[0-9.]+(?:[eE][+-]?[0-9]+)?$")
TRAIN_DELTA_RE = re.compile(r"\[dte-perf\]\[train-delta\] v(?P<version>\d+)\b")
TRAIN_STAGE_RE = re.compile(r"\[dte-perf\]\[train\] v(?P<version>\d+)\b")
INVERSION_RE = re.compile(r"\[dte-perf\]\[inversion\] v(?P<version>\d+)\b")
STEP_DIRTY_RE = re.compile(r"\[dte-perf\]\[step-dirty\]")
DIRTY_BIT_PROVIDER_RE = re.compile(r"\[dte-perf\]\[dirty-bit-provider\]")
STEP_DIRTY_PHASE_RE = re.compile(r"\bphase=(?P<phase>[A-Za-z_]+)\b")
INFER_RE = re.compile(r"\[dte-perf\]\[infer\] v(?P<version>\d+) rank (?P<rank>\d+)\b")
# awex emits bare "[perf]" (no dte coupling upstream); older logs used
# "[dte-perf]". Match both so historical trials stay analyzable.
AWEX_CHUNK_RE = re.compile(
    r"\[(?:dte-)?perf\]\[awex-chunk\].*?\btask=(?P<task>[^\s]+)\s+"
    r"chunk=(?P<chunk>\d+)/(?P<chunks>\d+)\b"
)
AWEX_RECURSIVE_RE = re.compile(
    r"\[(?:dte-)?perf\]\[awex-recursive\].*?\bstep=(?P<version>\d+)\b"
)
DELTA_BUILD_RE = re.compile(
    r"\[dte-perf\]\[delta-build\].*?\bstep=(?P<version>\d+) "
    r"phase=(?P<phase>[A-Za-z_]+)\b"
)
COLOCATE_DELTA_RE = re.compile(
    r"colocate delta v(?P<version>\d+) \[.*?\]: "
    r"changed (?P<changed>\d+)/(?P<total>\d+) "
    r"\((?P<changed_pct>[0-9.]+)%\) "
    r"sparse=(?P<sparse>\d+) dense_fallback=(?P<dense_fallback>\d+) "
    r"unchanged=(?P<unchanged>\d+) payload=(?P<payload_mb>[0-9.]+)MB "
    r"vs dense=(?P<dense_mb>[0-9.]+)MB"
)
KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)="
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    # Values may be followed by whitespace, a comma, an ANSI escape (colored
    # log lines end with e.g. "peak_mb=42786\x1b[0m"), or end-of-line.
    r"(?=\s|,|\x1b|$)"
)


SENDER_KEYS = (
    "compute_masks_ms",
    "encode_ms",
    "payload_mb",
    "dense_mb",
    "total_ms",
)
INVERSION_KEYS = (
    "zero_probe_ms",
    "alloc_mb",
    "peak_mb",
    "reconstruct_mcore_ms",
    "convert_hf_ms",
    "mask_loop_ms",
    "total_ms",
)
STEP_DIRTY_KEYS = (
    "capture_ms",
    "compare_ms",
    "pack_ms",
    "indices_ms",
    "snapshot_mb",
    "total_snapshot_mb",
    "bitset_mb",
    "indices_mb",
    "total_params",
    "captured_params",
    "skipped_by_cap",
    "captured_elements",
    "changed_elements",
    "indices_elements",
    "changed_ratio",
)
DIRTY_BIT_PROVIDER_KEYS = (
    "records",
    "complete",
    "collect_ms",
)
TRAIN_STAGE_KEYS = (
    "ipc_collect_ms",
    "sync_model_params_ms",
    "get_local_shard_params_ms",
    "delta_or_full_encode_ms",
    "capture_synced_state_ms",
    "release_grad_ms",
    "group_payload_tensors_ms",
    "release_weights_ms",
    "share_tensors_for_ipc_ms",
    "serialize_ipc_payload_ms",
    "put_serialized_payload_ms",
    "wait_inference_done_and_mark_synced_ms",
    "cleanup_ms",
    "alloc_mb",
    "peak_mb",
)
INFER_KEYS = (
    "wait_train_offloaded_ms",
    "wait_payload_kv_ms",
    "cuda_ipc_deserialize_ms",
    "resume_weights_ms",
    "apply_full_colocate_ms",
    "apply_decoded_delta_colocate_ms",
    "commit_empty_delta_ms",
    "commit_live_delta_ms",
    "cleanup_ms",
    "total_ms",
    "alloc_mb",
    "peak_mb",
)
AWEX_CHUNK_KEYS = (
    "chunk",
    "chunks",
    "send_peers",
    "recv_peers",
    "clone_mb",
    "total_ms",
)
AWEX_RECURSIVE_KEYS = (
    "rounds",
    "total_ms",
    "send_ops",
    "recv_ops",
)
DELTA_BUILD_KEYS = (
    "ops",
    "sparse_ops",
    "dense_ops",
    "empty_ops",
    "sparse_groups",
    "avg_ops_per_group",
    "max_ops_per_group",
    "sparse_input_nnz",
    "sparse_output_nnz",
    "sparse_zero_ops",
    "first_pass_ms",
    "dense_ms",
    "sparse_remap_ms",
    "dtype_cast_ms",
    "total_ms",
)
DELTA_SUMMARY_KEYS = (
    "changed_pct",
    "payload_mb",
    "dense_mb",
    "sparse",
    "dense_fallback",
    "unchanged",
)
DRIVER_STATS_KEY_MAP = {
    "timeperf/train_step": "train_step_s",
    "timeperf/update_weights": "update_weights_s",
    "ppo_actor/update/perf/optimizer_step_time": "optimizer_step_s",
    "ppo_actor/update/perf/step_dirty_capture_time": "step_dirty_capture_s",
    "ppo_actor/update/perf/step_dirty_compare_time": "step_dirty_compare_s",
    "ppo_actor/update/perf/step_dirty_pack_time": "step_dirty_pack_s",
    "ppo_actor/update/perf/step_dirty_indices_time": "step_dirty_indices_s",
    "ppo_actor/update/perf/step_dirty_snapshot_mb": "step_dirty_snapshot_mb",
    "ppo_actor/update/perf/step_dirty_bitset_mb": "step_dirty_bitset_mb",
    "ppo_actor/update/perf/step_dirty_indices_mb": "step_dirty_indices_mb",
    "ppo_actor/update/perf/dirty_bit_collect_time": "dirty_bit_collect_s",
    "ppo_actor/update/perf/dirty_bit_records": "dirty_bit_records",
    "ppo_actor/update/perf/dirty_bit_complete": "dirty_bit_complete",
}
DRIVER_STATS_KEYS = tuple(DRIVER_STATS_KEY_MAP.values())


def _read_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as fin:
        return list(fin)


def _parse_key_values(line: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for match in KEY_VALUE_RE.finditer(line):
        try:
            values[match.group("key")] = float(match.group("value"))
        except ValueError:
            continue
    return values


def _parse_stats_table_pairs(line: str) -> dict[str, float]:
    cells = [cell.strip() for cell in re.split(r"[│|]", line) if cell.strip()]
    pairs: dict[str, float] = {}
    for idx in range(len(cells) - 1):
        key = cells[idx]
        value = cells[idx + 1]
        if "/" not in key:
            continue
        if not NUMERIC_RE.match(value):
            continue
        try:
            pairs[key] = float(value)
        except ValueError:
            continue
    return pairs


def _append_metric(
    metrics: dict[int, dict[str, list[float]]],
    version: int,
    key: str,
    value: float,
) -> None:
    metrics[version][key].append(value)


def _series_summary(values: Iterable[float]) -> dict[str, float | int]:
    series = sorted(values)
    if not series:
        return {"count": 0}
    return {
        "count": len(series),
        "p50": statistics.median(series),
        "min": series[0],
        "max": series[-1],
    }


def _summarize_metrics(
    metrics: Mapping[int, Mapping[str, Sequence[float]]],
    keys: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for version in sorted(metrics):
        version_result: dict[str, dict[str, float | int]] = {}
        for key in keys:
            if key not in metrics[version]:
                continue
            version_result[key] = _series_summary(metrics[version][key])
        if version_result:
            result[str(version)] = version_result
    return result


def _summarize_phase_metrics(
    metrics: Mapping[int, Mapping[str, Mapping[str, Sequence[float]]]],
    keys: Sequence[str],
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    result: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for version in sorted(metrics):
        version_result: dict[str, dict[str, dict[str, float | int]]] = {}
        for phase in sorted(metrics[version]):
            phase_result: dict[str, dict[str, float | int]] = {}
            for key in keys:
                if key not in metrics[version][phase]:
                    continue
                phase_result[key] = _series_summary(metrics[version][phase][key])
            if phase_result:
                version_result[phase] = phase_result
        if version_result:
            result[str(version)] = version_result
    return result


def _summarize_global_metrics(
    metrics: Mapping[str, Sequence[float]],
    keys: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for key in keys:
        if key in metrics:
            result[key] = _series_summary(metrics[key])
    return result


def _parse_driver(
    lines: Sequence[str],
) -> tuple[dict[int, float], dict[int, float], dict[str, Any]]:
    gateway_ms: dict[int, float] = {}
    trainer_update_s: dict[int, float] = {}
    driver_stats: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pending_update_version: int | None = None
    current_stats_version: int | None = None

    for line in lines:
        gateway_match = GATEWAY_RE.search(line)
        if gateway_match:
            version = int(gateway_match.group("version"))
            gateway_ms[version] = float(gateway_match.group("duration_ms"))
            pending_update_version = version
            current_stats_version = version
            continue

        if current_stats_version is not None and ("│" in line or "|" in line):
            for raw_key, value in _parse_stats_table_pairs(line).items():
                key = DRIVER_STATS_KEY_MAP.get(raw_key)
                if key is not None:
                    _append_metric(driver_stats, current_stats_version, key, value)

        if pending_update_version is not None and "timeperf/update_weights" in line:
            update_match = TRAINER_UPDATE_RE.search(line)
            if update_match:
                trainer_update_s[pending_update_version] = float(
                    update_match.group("value")
                )
                pending_update_version = None

    return (
        gateway_ms,
        trainer_update_s,
        _summarize_metrics(driver_stats, DRIVER_STATS_KEYS),
    )


def _parse_train(lines: Sequence[str]) -> dict[str, Any]:
    train_delta: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    inversion: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    train_stage: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    step_dirty: dict[str, list[float]] = defaultdict(list)
    dirty_bit_provider: dict[str, list[float]] = defaultdict(list)
    step_dirty_by_round: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    delta_summary: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    step_dirty_round = 0
    step_dirty_seen_after = False

    for line in lines:
        summary_match = COLOCATE_DELTA_RE.search(line)
        if summary_match:
            version = int(summary_match.group("version"))
            for key in DELTA_SUMMARY_KEYS:
                _append_metric(
                    delta_summary, version, key, float(summary_match.group(key))
                )

        train_delta_match = TRAIN_DELTA_RE.search(line)
        if train_delta_match:
            version = int(train_delta_match.group("version"))
            values = _parse_key_values(line)
            for key in SENDER_KEYS:
                if key in values:
                    _append_metric(train_delta, version, key, values[key])

        inversion_match = INVERSION_RE.search(line)
        if inversion_match:
            version = int(inversion_match.group("version"))
            values = _parse_key_values(line)
            for key in INVERSION_KEYS:
                if key in values:
                    _append_metric(inversion, version, key, values[key])

        train_stage_match = TRAIN_STAGE_RE.search(line)
        if train_stage_match:
            version = int(train_stage_match.group("version"))
            values = _parse_key_values(line)
            for key in TRAIN_STAGE_KEYS:
                if key in values:
                    _append_metric(train_stage, version, key, values[key])

        if STEP_DIRTY_RE.search(line):
            phase_match = STEP_DIRTY_PHASE_RE.search(line)
            phase = phase_match.group("phase") if phase_match else ""
            if phase == "before" and step_dirty_seen_after:
                step_dirty_round += 1
                step_dirty_seen_after = False
            elif step_dirty_round == 0:
                step_dirty_round = 1
            if phase == "after":
                step_dirty_seen_after = True

            values = _parse_key_values(line)
            for key in STEP_DIRTY_KEYS:
                if key in values:
                    step_dirty[key].append(values[key])
                    step_dirty_by_round[step_dirty_round][key].append(values[key])

        if DIRTY_BIT_PROVIDER_RE.search(line):
            values = _parse_key_values(line)
            for key in DIRTY_BIT_PROVIDER_KEYS:
                if key in values:
                    dirty_bit_provider[key].append(values[key])

    return {
        "delta_summary": _summarize_metrics(delta_summary, DELTA_SUMMARY_KEYS),
        "train_delta": _summarize_metrics(train_delta, SENDER_KEYS),
        "inversion": _summarize_metrics(inversion, INVERSION_KEYS),
        "train_stage": _summarize_metrics(train_stage, TRAIN_STAGE_KEYS),
        "step_dirty": _summarize_global_metrics(step_dirty, STEP_DIRTY_KEYS),
        "dirty_bit_provider": _summarize_global_metrics(
            dirty_bit_provider,
            DIRTY_BIT_PROVIDER_KEYS,
        ),
        "step_dirty_by_round": {
            str(round_id): _summarize_global_metrics(round_metrics, STEP_DIRTY_KEYS)
            for round_id, round_metrics in sorted(step_dirty_by_round.items())
        },
    }


def _parse_infer(lines: Sequence[str]) -> dict[str, Any]:
    # Infer logs print one stage per line; aggregate by rank first so each rank
    # contributes at most once to the p50 for each stage.
    by_rank: dict[int, dict[int, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    delta_build: dict[int, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    awex_chunk: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    awex_recursive: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for line in lines:
        awex_chunk_match = AWEX_CHUNK_RE.search(line)
        if awex_chunk_match:
            task = awex_chunk_match.group("task")
            _, separator, raw_version = task.rpartition("-")
            if separator and raw_version.isdigit():
                version = int(raw_version)
                values = _parse_key_values(line)
                values["chunk"] = float(awex_chunk_match.group("chunk"))
                values["chunks"] = float(awex_chunk_match.group("chunks"))
                for key in AWEX_CHUNK_KEYS:
                    if key in values:
                        _append_metric(awex_chunk, version, key, values[key])

        awex_recursive_match = AWEX_RECURSIVE_RE.search(line)
        if awex_recursive_match:
            values = _parse_key_values(line)
            if "total_ms" in values:
                version = int(awex_recursive_match.group("version"))
                for key in AWEX_RECURSIVE_KEYS:
                    if key in values:
                        _append_metric(awex_recursive, version, key, values[key])

        delta_build_match = DELTA_BUILD_RE.search(line)
        if delta_build_match:
            version = int(delta_build_match.group("version"))
            phase = delta_build_match.group("phase")
            values = _parse_key_values(line)
            for key in DELTA_BUILD_KEYS:
                if key in values:
                    delta_build[version][phase][key].append(values[key])

        infer_match = INFER_RE.search(line)
        if not infer_match:
            continue
        version = int(infer_match.group("version"))
        rank = int(infer_match.group("rank"))
        by_rank[version][rank].update(_parse_key_values(line))

    metrics: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for version, rank_values in by_rank.items():
        for values in rank_values.values():
            for key in INFER_KEYS:
                if key in values:
                    _append_metric(metrics, version, key, values[key])

    return {
        "infer": _summarize_metrics(metrics, INFER_KEYS),
        "awex_chunk": _summarize_metrics(awex_chunk, AWEX_CHUNK_KEYS),
        "awex_recursive": _summarize_metrics(awex_recursive, AWEX_RECURSIVE_KEYS),
        "delta_build": _summarize_phase_metrics(delta_build, DELTA_BUILD_KEYS),
    }


def parse_sync_perf_logs(
    driver_log: Path | None = None,
    train_log: Path | None = None,
    infer_log: Path | None = None,
) -> dict[str, Any]:
    """Parse DTE sync timing logs into per-version summary statistics."""

    gateway_ms, trainer_update_s, driver_stats = _parse_driver(_read_lines(driver_log))
    train_result = _parse_train(_read_lines(train_log))
    infer_result = _parse_infer(_read_lines(infer_log))

    return {
        "gateway_ms": {str(k): v for k, v in sorted(gateway_ms.items())},
        "trainer_update_s": {str(k): v for k, v in sorted(trainer_update_s.items())},
        "driver_stats": driver_stats,
        **train_result,
        **infer_result,
    }


def _fmt_seconds(summary: Mapping[str, float | int] | None) -> str:
    if not summary or "p50" not in summary:
        return "-"
    p50 = float(summary["p50"]) / 1000.0
    min_value = float(summary["min"]) / 1000.0
    max_value = float(summary["max"]) / 1000.0
    return f"{p50:.3f}s [{min_value:.3f}, {max_value:.3f}]"


def _fmt_value(summary: Mapping[str, float | int] | None, suffix: str = "") -> str:
    if not summary or "p50" not in summary:
        return "-"
    p50 = float(summary["p50"])
    min_value = float(summary["min"])
    max_value = float(summary["max"])
    return f"{p50:.3f}{suffix} [{min_value:.3f}, {max_value:.3f}]"


def _fmt_seconds_value(summary: Mapping[str, float | int] | None) -> str:
    if not summary or "p50" not in summary:
        return "-"
    p50 = float(summary["p50"])
    min_value = float(summary["min"])
    max_value = float(summary["max"])
    return f"{p50:.3f}s [{min_value:.3f}, {max_value:.3f}]"


def _fmt_millions(summary: Mapping[str, float | int] | None) -> str:
    if not summary or "p50" not in summary:
        return "-"
    p50 = float(summary["p50"]) / 1_000_000.0
    min_value = float(summary["min"]) / 1_000_000.0
    max_value = float(summary["max"]) / 1_000_000.0
    return f"{p50:.1f}M [{min_value:.1f}, {max_value:.1f}]"


def _versions(summary: Mapping[str, Any]) -> list[str]:
    versions: set[str] = set(summary.get("gateway_ms", {}))
    for section in (
        "delta_summary",
        "train_delta",
        "inversion",
        "driver_stats",
        "train_stage",
        "infer",
        "awex_chunk",
        "awex_recursive",
        "delta_build",
    ):
        versions.update(summary.get(section, {}))
    return sorted(versions, key=lambda item: int(item))


def format_markdown_summary(summary: Mapping[str, Any]) -> str:
    """Format parsed DTE timing summaries as compact markdown tables."""

    lines: list[str] = ["# DTE Sync Perf Summary", ""]
    lines.extend(
        [
            "| version | Gateway | trainer update | changed | payload/rank | sender compute | encode | receiver full apply | receiver delta apply |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for version in _versions(summary):
        gateway = summary.get("gateway_ms", {}).get(version)
        trainer = summary.get("trainer_update_s", {}).get(version)
        delta = summary.get("delta_summary", {}).get(version, {})
        train_delta = summary.get("train_delta", {}).get(version, {})
        infer = summary.get("infer", {}).get(version, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    f"v{version}",
                    "-" if gateway is None else f"{gateway / 1000.0:.3f}s",
                    "-" if trainer is None else f"{trainer:.3f}s",
                    _fmt_value(delta.get("changed_pct"), "%"),
                    _fmt_value(delta.get("payload_mb"), "MB"),
                    _fmt_seconds(train_delta.get("compute_masks_ms")),
                    _fmt_seconds(train_delta.get("encode_ms")),
                    _fmt_seconds(infer.get("apply_full_colocate_ms")),
                    _fmt_seconds(infer.get("apply_decoded_delta_colocate_ms")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Sender Inversion",
            "",
            "| version | zero_probe | reconstruct_mcore | convert_hf | mask_loop | inversion total |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for version in _versions(summary):
        inversion = summary.get("inversion", {}).get(version, {})
        if not inversion:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"v{version}",
                    _fmt_seconds(inversion.get("zero_probe_ms")),
                    _fmt_seconds(inversion.get("reconstruct_mcore_ms")),
                    _fmt_seconds(inversion.get("convert_hf_ms")),
                    _fmt_seconds(inversion.get("mask_loop_ms")),
                    _fmt_seconds(inversion.get("total_ms")),
                ]
            )
            + " |"
        )

    driver_stats = summary.get("driver_stats", {})
    if driver_stats:
        lines.extend(
            [
                "",
                "## Trainer Stats",
                "",
                "| version | train_step | update_weights | optimizer_step | step_dirty_capture | step_dirty_compare | step_dirty_pack | step_dirty_indices |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for version in _versions(summary):
            stats = driver_stats.get(version, {})
            if not stats:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"v{version}",
                        _fmt_seconds_value(stats.get("train_step_s")),
                        _fmt_seconds_value(stats.get("update_weights_s")),
                        _fmt_seconds_value(stats.get("optimizer_step_s")),
                        _fmt_seconds_value(stats.get("step_dirty_capture_s")),
                        _fmt_seconds_value(stats.get("step_dirty_compare_s")),
                        _fmt_seconds_value(stats.get("step_dirty_pack_s")),
                        _fmt_seconds_value(stats.get("step_dirty_indices_s")),
                    ]
                )
                + " |"
            )

    step_dirty = summary.get("step_dirty", {})
    if step_dirty:
        lines.extend(
            [
                "",
                "## Optimizer Step Dirty Dry-Run",
                "",
                "| capture | compare | pack | indices | snapshot | bitset | indices bytes | captured params | changed ratio |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| "
                + " | ".join(
                    [
                        _fmt_seconds(step_dirty.get("capture_ms")),
                        _fmt_seconds(step_dirty.get("compare_ms")),
                        _fmt_seconds(step_dirty.get("pack_ms")),
                        _fmt_seconds(step_dirty.get("indices_ms")),
                        _fmt_value(step_dirty.get("snapshot_mb"), "MB"),
                        _fmt_value(step_dirty.get("bitset_mb"), "MB"),
                        _fmt_value(step_dirty.get("indices_mb"), "MB"),
                        _fmt_value(step_dirty.get("captured_params")),
                        _fmt_value(step_dirty.get("changed_ratio")),
                    ]
                )
                + " |",
            ]
        )

    dirty_bit_provider = summary.get("dirty_bit_provider", {})
    if dirty_bit_provider:
        lines.extend(
            [
                "",
                "## Dirty-Bit Provider",
                "",
                "| collect | records | complete |",
                "|---:|---:|---:|",
                "| "
                + " | ".join(
                    [
                        _fmt_seconds(dirty_bit_provider.get("collect_ms")),
                        _fmt_value(dirty_bit_provider.get("records")),
                        _fmt_value(dirty_bit_provider.get("complete")),
                    ]
                )
                + " |",
            ]
        )

    train_stage = summary.get("train_stage", {})
    if train_stage:
        lines.extend(
            [
                "",
                "## Training Sender Stages",
                "",
                "| version | sync_model_params | encode | release_grad | group_payload | release_weights | serialize | wait_infer | cleanup |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for version in _versions(summary):
            stages = train_stage.get(version, {})
            if not stages:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"v{version}",
                        _fmt_seconds(stages.get("sync_model_params_ms")),
                        _fmt_seconds(stages.get("delta_or_full_encode_ms")),
                        _fmt_seconds(stages.get("release_grad_ms")),
                        _fmt_seconds(stages.get("group_payload_tensors_ms")),
                        _fmt_seconds(stages.get("release_weights_ms")),
                        _fmt_seconds(stages.get("serialize_ipc_payload_ms")),
                        _fmt_seconds(
                            stages.get("wait_inference_done_and_mark_synced_ms")
                        ),
                        _fmt_seconds(stages.get("cleanup_ms")),
                    ]
                )
                + " |"
            )

    step_dirty_by_round = summary.get("step_dirty_by_round", {})
    if step_dirty_by_round:
        lines.extend(
            [
                "",
                "## Optimizer Step Dirty Dry-Run By Round",
                "",
                "| round | capture | compare | pack | indices | snapshot | bitset | indices bytes | changed ratio |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for round_id in sorted(step_dirty_by_round, key=lambda item: int(item)):
            round_summary = step_dirty_by_round[round_id]
            lines.append(
                "| "
                + " | ".join(
                    [
                        round_id,
                        _fmt_seconds(round_summary.get("capture_ms")),
                        _fmt_seconds(round_summary.get("compare_ms")),
                        _fmt_seconds(round_summary.get("pack_ms")),
                        _fmt_seconds(round_summary.get("indices_ms")),
                        _fmt_value(round_summary.get("snapshot_mb"), "MB"),
                        _fmt_value(round_summary.get("bitset_mb"), "MB"),
                        _fmt_value(round_summary.get("indices_mb"), "MB"),
                        _fmt_value(round_summary.get("changed_ratio")),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Inference Receiver",
            "",
            "| version | wait_train_offloaded | resume_weights | apply_full | apply_decoded_delta | commit_empty_delta | cleanup | infer total |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for version in _versions(summary):
        infer = summary.get("infer", {}).get(version, {})
        if not infer:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"v{version}",
                    _fmt_seconds(infer.get("wait_train_offloaded_ms")),
                    _fmt_seconds(infer.get("resume_weights_ms")),
                    _fmt_seconds(infer.get("apply_full_colocate_ms")),
                    _fmt_seconds(infer.get("apply_decoded_delta_colocate_ms")),
                    _fmt_seconds(infer.get("commit_empty_delta_ms")),
                    _fmt_seconds(infer.get("cleanup_ms")),
                    _fmt_seconds(infer.get("total_ms")),
                ]
            )
            + " |"
        )

    awex_chunk = summary.get("awex_chunk", {})
    awex_recursive = summary.get("awex_recursive", {})
    if awex_chunk or awex_recursive:
        lines.extend(
            [
                "",
                "## AWEX P2P",
                "",
                "| version | recursive total | send ops | recv ops | chunk total | chunk clone | send peers | recv peers | chunks observed |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for version in _versions(summary):
            recursive = awex_recursive.get(version, {})
            chunk = awex_chunk.get(version, {})
            if not recursive and not chunk:
                continue
            chunk_total = chunk.get("total_ms")
            chunk_count = chunk_total.get("count") if chunk_total else 0
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"v{version}",
                        _fmt_seconds(recursive.get("total_ms")),
                        _fmt_value(recursive.get("send_ops")),
                        _fmt_value(recursive.get("recv_ops")),
                        _fmt_seconds(chunk.get("total_ms")),
                        _fmt_value(chunk.get("clone_mb"), "MB"),
                        _fmt_value(chunk.get("send_peers")),
                        _fmt_value(chunk.get("recv_peers")),
                        str(chunk_count),
                    ]
                )
                + " |"
            )

    delta_build = summary.get("delta_build", {})
    if delta_build:
        lines.extend(
            [
                "",
                "## Receiver Delta Build",
                "",
                "| version | phase | input nnz | output nnz | sparse ops | groups | avg ops/group | sparse remap | build total |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for version in _versions(summary):
            phase_metrics = delta_build.get(version, {})
            for phase in sorted(phase_metrics):
                metrics = phase_metrics[phase]
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"v{version}",
                            phase,
                            _fmt_millions(metrics.get("sparse_input_nnz")),
                            _fmt_millions(metrics.get("sparse_output_nnz")),
                            _fmt_value(metrics.get("sparse_ops")),
                            _fmt_value(metrics.get("sparse_groups")),
                            _fmt_value(metrics.get("avg_ops_per_group")),
                            _fmt_seconds(metrics.get("sparse_remap_ms")),
                            _fmt_seconds(metrics.get("total_ms")),
                        ]
                    )
                    + " |"
                )

    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize DTE weight-sync timing logs."
    )
    parser.add_argument("--driver-log", type=Path, help="Top-level Slurm driver log.")
    parser.add_argument("--train-log", type=Path, help="AWEX train worker log.")
    parser.add_argument("--infer-log", type=Path, help="AWEX inference server log.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    summary = parse_sync_perf_logs(
        driver_log=args.driver_log,
        train_log=args.train_log,
        infer_log=args.infer_log,
    )
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_markdown_summary(summary), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
