# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

TIMESTAMP_COLUMNS = [
    "submit_ts",
    "finalized_ts",
]
DURATION_COLUMNS = [
    "total_s",
    "generate_s",
    "reward_s",
    "toolcall_s",
]
DEFAULT_PHASE_ORDER: tuple[str, ...] = ("generate", "reward", "toolcall")
STATUS_COLORS: dict[str, str] = {
    "accepted": "#1f77b4",
    "rejected": "#d62728",
    "failed": "#9467bd",
    "dropped": "#ff7f0e",
}
SEGMENT_STYLES: dict[str, dict[str, Any]] = {
    "idle": {"label": "Idle", "color": "#d1d5db"},  # Medium gray for idle periods
    "generate": {"label": "Generate", "color": "#2563eb"},  # Blue for generation
    "reward": {"label": "Reward", "color": "#dc2626"},  # Red for reward
    "toolcall": {"label": "Tool Call", "color": "#f59e0b"},  # Orange for tool calls
}

SEGMENT_OUTLINE = {"color": "#1f1f1f", "width": 0.7}


def _compute_bar_width(num_sessions: int) -> float:
    """Shrink bars when many sessions are rendered to avoid cluttering visuals."""
    if num_sessions <= 0:
        return 0.95
    # Increased bar width for better visibility: minimum 0.5, maximum 0.95
    return float(max(0.5, min(0.95, 60.0 / num_sessions)))


HISTOGRAM_METRICS: dict[str, dict[str, Any]] = {
    "total_s": {"label": "Total duration", "color": "#8b5cf6"},
    "generate_s": {"label": "Generation duration", "color": "#3b82f6"},
    "reward_s": {"label": "Reward calculation duration", "color": "#10b981"},
    "toolcall_s": {"label": "Tool call duration", "color": "#f59e0b"},
}

MIN_STEP_GAP = 0.5


def _compute_histogram_bins(
    values: pd.Series, *, max_bins: int = 120, min_bins: int = 0
) -> dict[str, float] | None:
    """Derive x-axis bin settings to avoid overly coarse histograms."""
    if values.empty:
        return {
            "start": 0.0,
            "end": 1.0,
            "size": 0.1,
        }
    cleaned = values.dropna()
    min_bins = max(0, min_bins)

    def _isfinite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    cleaned = cleaned[cleaned.apply(_isfinite)]
    if cleaned.empty:
        return {
            "start": 0.0,
            "end": 1.0,
            "size": 0.1,
        }
    vmin = float(cleaned.min())
    vmax = float(cleaned.max())
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return {
            "start": 0.0,
            "end": 1.0,
            "size": 0.1,
        }
    span = vmax - vmin
    count = len(cleaned)
    if count <= 1:
        span = max(span, max(abs(vmin), abs(vmax), 1.0) * 0.05)
        width = max(span / 8.0, 1e-3)
        return {
            "start": vmin - width * 0.5,
            "end": vmin + width * 1.5,
            "size": width,
        }

    q1, q3 = cleaned.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    if math.isfinite(iqr) and iqr > 0:
        width = 2.0 * iqr / (count ** (1.0 / 3.0))
    else:
        width = 0.0
    if not math.isfinite(width) or width <= 0:
        target_bins = min(max_bins, max(40, count // 2))
        width = span / max(target_bins, 1)
    min_width = max(span / max_bins if span > 0 else 0.0, 1e-3)
    width = max(width, min_width)
    if min_bins > 0 and span > 0:
        target_width = span / float(max(min_bins, 1))
        width = min(width, max(target_width, 1e-3))
        width = max(width, min_width)
    return {
        "start": vmin,
        "end": vmax + width,
        "size": width,
    }


def _format_index_label(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return str(int(numeric))


def _resolve_record_files(source: str | Path) -> list[Path]:
    """Resolve an input path, directory, or glob into concrete JSONL files."""
    source_path = Path(source).expanduser()
    if source_path.exists():
        if source_path.is_file():
            return [source_path]
        if source_path.is_dir():
            return sorted(p for p in source_path.glob("*.jsonl") if p.is_file())
    matches = [Path(p).expanduser() for p in glob(str(source_path), recursive=True)]
    return sorted(p for p in matches if p.is_file())


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for lineno, raw_line in enumerate(fin, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(
                    f"Failed to parse JSON on line {lineno} of {path}: {exc}"
                ) from exc

            records.append(record)
    return records


def _ensure_numeric(df: pd.DataFrame) -> None:
    for column in TIMESTAMP_COLUMNS + ["rank", "task_id", "session_id"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in DURATION_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")


def _maybe_compute_durations(df: pd.DataFrame) -> None:
    """Compute derived duration columns if not present.

    For new session format, durations are pre-computed in the trace data.
    This function is kept for backward compatibility.
    """
    # New format already has these computed, but check anyway
    if (
        "total_s" not in df.columns
        and "submit_ts" in df.columns
        and "finalized_ts" in df.columns
    ):
        df["total_s"] = df["finalized_ts"] - df["submit_ts"]


def _extract_phase_timestamps(df: pd.DataFrame) -> None:
    """Extract phase timestamps from nested phases structure.

    Converts the nested phases dict structure into flattened columns for easier
    processing. For each phase (generate, reward, toolcall), extracts all execution spans
    and adds them as additional data that can be used for visualization.

    The phases field format from SessionRecord.to_dict():
    {
        "phases": {
            "generate": [{"start_ts": 1.5, "end_ts": 3.0}, ...],
            "reward": [{"start_ts": 4.3, "end_ts": 5.3}, ...],
            "toolcall": [{"start_ts": 6.0, "end_ts": 7.5}, ...]
        }
    }

    This function adds a '_phases_data' column containing the parsed structure
    for use in timeline visualization.
    """

    def parse_phases(phases_data):
        if phases_data is None or (
            isinstance(phases_data, float) and math.isnan(phases_data)
        ):
            return None
        if not isinstance(phases_data, dict):
            return None
        return phases_data

    if "phases" in df.columns:
        df["_phases_data"] = df["phases"].apply(parse_phases)
    else:
        df["_phases_data"] = None


def _compute_offsets(df: pd.DataFrame) -> float:
    """Compute time offsets relative to earliest submit timestamp."""
    base_time = (
        float(df["submit_ts"].min(skipna=True)) if "submit_ts" in df else math.nan
    )
    if not math.isfinite(base_time):
        base_time = 0.0
    for column in TIMESTAMP_COLUMNS:
        if column in df.columns:
            df[f"{column}_offset"] = df[column] - base_time
    return base_time


def _determine_step_timepoints(
    df: pd.DataFrame,
    consumer_batch_size: int,
) -> tuple[list[float], str | None]:
    """Determine step boundaries based on per-rank task completion with global synchronization.

    Args:
        df: DataFrame with columns: rank, task_id, finalized_ts_offset, status
        consumer_batch_size: Global batch size (total tasks across all ranks per step)

    Returns:
        timepoints: List of global step boundary timestamps
        method: Description of the detection method used
    """
    # Validate required columns
    required_cols = ["rank", "task_id", "finalized_ts_offset", "status"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return [], f"Missing required columns: {missing}"

    # Filter valid data: must have all required fields
    valid_df = df.dropna(subset=required_cols).copy()
    if valid_df.empty:
        return [], "No valid data with rank, task_id, finalized_ts_offset, and status"

    # Report total sessions before filtering
    total_sessions = len(valid_df)

    # Only keep accepted sessions (rejected sessions are not consumed)
    valid_df = valid_df[valid_df["status"] == "accepted"]
    accepted_sessions = len(valid_df)

    if valid_df.empty:
        return (
            [],
            f"No accepted sessions found (total sessions: {total_sessions}, all rejected)",
        )

    # Report filtering results
    rejected_count = total_sessions - accepted_sessions
    if rejected_count > 0:
        print(
            f"Filtered out {rejected_count} rejected session(s) "
            f"({accepted_sessions} accepted, {rejected_count} rejected)"
        )
    # Calculate number of ranks
    ranks = sorted(valid_df["rank"].unique())
    num_ranks = len(ranks)

    if num_ranks == 0:
        return [], "No ranks found"

    # Calculate per-rank quota
    per_rank_quota = consumer_batch_size // num_ranks
    if per_rank_quota < 1:
        return (
            [],
            f"consumer_batch_size ({consumer_batch_size}) < num_ranks ({num_ranks}). Increase batch size or reduce ranks.",
        )

    # For each rank, compute task completion times and step boundaries
    rank_step_times: dict[Any, list[float]] = {}

    for rank in ranks:
        rank_df = valid_df[valid_df["rank"] == rank]

        # Group by task_id and find the slowest (max) finalized_ts_offset for each task
        task_completion_times = (
            rank_df.groupby("task_id")["finalized_ts_offset"].max().sort_values()
        )

        if task_completion_times.empty:
            rank_step_times[rank] = []
            continue

        # Split tasks into steps based on per_rank_quota
        step_boundaries: list[float] = []
        task_times = task_completion_times.values

        # Only form complete steps (discard incomplete tail)
        num_complete_steps = len(task_times) // per_rank_quota

        for step_idx in range(num_complete_steps):
            # The end of this step is the completion time of the last task in the batch
            end_task_idx = (step_idx + 1) * per_rank_quota - 1
            step_end_time = float(task_times[end_task_idx])
            step_boundaries.append(step_end_time)

        rank_step_times[rank] = step_boundaries

    # Global synchronization: find minimum number of steps across all ranks
    if not rank_step_times:
        return [], "No rank step boundaries computed"

    min_steps = min(len(times) for times in rank_step_times.values())

    if min_steps == 0:
        return [], "No complete steps formed (insufficient tasks per rank)"

    # For each global step, take the maximum time across all ranks
    global_timepoints: list[float] = []
    for step_idx in range(min_steps):
        step_times = [rank_step_times[rank][step_idx] for rank in ranks]
        global_step_time = max(step_times)
        global_timepoints.append(global_step_time)

    method = (
        f"consumer_batch_size={consumer_batch_size}, "
        f"{num_ranks} ranks, "
        f"{per_rank_quota} tasks/rank/step"
    )

    return global_timepoints, method


def _apply_step_assignments(
    df: pd.DataFrame, step_timepoints: Sequence[float] | None
) -> pd.Series:
    """Assign step IDs to sessions based on global step timepoints.

    Args:
        df: DataFrame with finalized_ts_offset column
        step_timepoints: List of step boundary timestamps (sessions <= timepoint belong to that step)

    Returns:
        Series with counts of sessions per step
    """
    if not step_timepoints:
        df["step_id"] = pd.NA
        return pd.Series(dtype="int64")

    offsets = df.get("finalized_ts_offset")
    if offsets is None:
        df["step_id"] = pd.NA
        return pd.Series(dtype="int64")

    valid_offsets = offsets.dropna()
    if valid_offsets.empty:
        df["step_id"] = pd.NA
        return pd.Series(dtype="int64")

    # Initialize all as NA
    df["step_id"] = pd.NA

    # Prepare timepoints as a sorted, finite numpy array
    tp = np.array(step_timepoints, dtype=float)
    if tp.size == 0 or not np.isfinite(tp).all():
        df["step_id"] = pd.NA
        return pd.Series(dtype="int64")
    tp_sorted = np.unique(tp)

    # Build bins such that intervals are (-inf, t1], (t1, t2], ..., (t_{n-1}, t_n]
    bins = np.concatenate(([-np.inf], tp_sorted))
    labels = list(range(len(tp_sorted)))

    # Use pd.cut to assign each finalized offset to the first step with timepoint >= offset.
    # Values beyond the last timepoint (offset > last timepoint) will be NaN (unassigned), matching previous logic.
    step_series = pd.cut(offsets, bins=bins, labels=labels, right=True)
    df["step_id"] = step_series.astype("Int64")

    # Count sessions per step (0..N-1)
    counts_series = (
        df["step_id"]
        .value_counts()
        .reindex(range(len(tp_sorted)), fill_value=0)
        .sort_index()
        .astype("int64")
    )
    return counts_series


def _build_timeline(
    fig: go.Figure, df: pd.DataFrame, *, base_time: float, row: int = 1, col: int = 1
) -> None:
    """Build timeline visualization showing session lifecycle with phases.

    For the new session format, displays:
    - Submit to finalized span (full session duration)
    - Individual phase executions (generate, reward, toolcall) within the span
    - Idle periods (gaps between phases) in light gray

    Args:
        fig: Plotly figure to add traces to
        df: DataFrame containing session records
        base_time: Base timestamp for offset calculation (should be min submit_ts across all data)
        row: Subplot row number
        col: Subplot column number
    """
    legend_seen: set[str] = set()
    timeline_df = df.dropna(subset=["submit_ts"]).copy()
    timeline_df = timeline_df.sort_values("submit_ts")
    bar_width = _compute_bar_width(len(timeline_df))

    for idx, record_tuple in enumerate(timeline_df.itertuples()):
        rank_label = _format_index_label(getattr(record_tuple, "rank", math.nan))
        role_value = getattr(record_tuple, "role", None)
        task_label = _format_index_label(getattr(record_tuple, "task_id", math.nan))
        session_label = _format_index_label(
            getattr(record_tuple, "session_id", math.nan)
        )
        if role_value:
            label = f"[{role_value}]r{rank_label}-t{task_label}-s{session_label}"
        else:
            label = f"r{rank_label}-t{task_label}-s{session_label}"

        submit_offset = getattr(record_tuple, "submit_ts_offset", math.nan)
        finalized_offset = getattr(record_tuple, "finalized_ts_offset", math.nan)

        if not math.isfinite(submit_offset):
            continue

        # Get phases data from the record
        record_idx = record_tuple.Index
        phases_data = None
        if record_idx in timeline_df.index:
            original_record = timeline_df.loc[record_idx]
            phases_data = original_record.get("_phases_data")

        # Collect all phase spans with their timestamps
        # (start_offset, end_offset, phase_name)
        all_spans: list[tuple[float, float, str]] = []

        if isinstance(phases_data, dict):
            default_phases = set(DEFAULT_PHASE_ORDER)
            additional_phases = sorted(
                p for p in phases_data if str(p) not in default_phases
            )
            phase_sequence = list(DEFAULT_PHASE_ORDER) + additional_phases

            for phase_name in phase_sequence:
                phase_executions = phases_data.get(phase_name, [])
                if not isinstance(phase_executions, list):
                    continue

                for execution in phase_executions:
                    if not isinstance(execution, dict):
                        continue
                    start_ts = execution.get("start_ts")
                    end_ts = execution.get("end_ts")
                    if start_ts is not None and end_ts is not None:
                        # All timestamps should use the same base_time reference
                        start_offset = start_ts - base_time
                        end_offset = end_ts - base_time
                        all_spans.append((start_offset, end_offset, phase_name))

        # Sort spans by start time
        all_spans.sort(key=lambda x: x[0])

        # Helper function to add a bar segment
        def add_segment(
            start: float,
            duration: float,
            segment_key: str,
            segment_label: str | None = None,
        ) -> None:
            if duration <= 0:
                return

            style = SEGMENT_STYLES.get(segment_key, {})
            color = style.get("color", "#cccccc")
            label_text = segment_label or style.get("label", segment_key)

            trace_name = label_text if label_text not in legend_seen else None

            # Calculate end offset for display
            end_offset = start + duration

            # Build hover template with step info
            hover_lines = [
                f"Session: {label}",
                f"Segment: {label_text}",
            ]
            hover_lines.extend(
                [
                    f"Start offset: {start:.3f}s",
                    f"End offset: {end_offset:.3f}s",
                    f"Duration: {duration:.3f}s",
                ]
            )
            hover_text = "<br>".join(hover_lines) + "<extra></extra>"

            fig.add_trace(
                go.Bar(
                    x=[duration],
                    y=[label],
                    base=start,
                    orientation="h",
                    marker=dict(
                        color=color,
                        opacity=0.9 if segment_key != "idle" else 0.3,
                        line=SEGMENT_OUTLINE if segment_key != "idle" else {},
                    ),
                    width=bar_width,
                    name=trace_name,
                    hovertemplate=hover_text,
                    showlegend=trace_name is not None,
                ),
                row=row,
                col=col,
            )
            if trace_name:
                legend_seen.add(label_text)

        # Build the timeline: fill the entire submit->finalized span
        current_time = submit_offset

        # Add idle from submit to first phase (if any)
        if all_spans:
            first_phase_start = all_spans[0][0]
            gap_duration = first_phase_start - current_time
            if gap_duration > 1e-6:  # Only add gap if meaningful
                add_segment(current_time, gap_duration, "idle", "Idle")
                current_time = first_phase_start

        # Add all phases and gaps between them
        for i, (phase_start, phase_end, phase_name) in enumerate(all_spans):
            # Add gap before this phase if there is one
            gap_duration = phase_start - current_time
            if gap_duration > 1e-6:  # Only add gap if it's meaningful (> 1 microsecond)
                add_segment(current_time, gap_duration, "idle", "Idle")

            # Add the phase segment
            phase_duration = phase_end - phase_start
            add_segment(phase_start, phase_duration, phase_name)
            current_time = phase_end

        # Add idle from last phase to finalized (if finalized exists and is after last phase)
        if math.isfinite(finalized_offset) and finalized_offset > current_time:
            gap_duration = finalized_offset - current_time
            if gap_duration > 1e-6:
                add_segment(current_time, gap_duration, "idle", "Idle")

    fig.update_yaxes(autorange="reversed", title="Session", row=row, col=col)
    fig.update_xaxes(title="Time offset (s)", row=row, col=col)


def _build_latency_scatter(
    fig: go.Figure, df: pd.DataFrame, *, row: int = 1, col: int = 1
) -> dict[str, float]:
    """Build scatter plot showing total session duration over time."""
    scatter_df = df.dropna(subset=["finalized_ts_offset", "total_s"]).copy()
    if scatter_df.empty:
        return {}

    # Check for NA ranks and raise error
    if scatter_df["rank"].isna().any():
        na_count = scatter_df["rank"].isna().sum()
        raise ValueError(
            f"Found {na_count} session(s) with NA rank in latency data. "
            "All sessions must have a valid rank value."
        )

    p95 = float(scatter_df["total_s"].quantile(0.95))
    p50 = float(scatter_df["total_s"].quantile(0.5))
    scatter_df["is_anomaly"] = scatter_df["total_s"] > p95
    palette = qualitative.Plotly or ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ranked_values = scatter_df["rank"].dropna().unique().tolist()
    rank_colors: dict[Any, str] = {}
    for idx, value in enumerate(sorted(ranked_values)):
        rank_colors[value] = palette[idx % len(palette)]

    def _subset_rank_label(value: Any) -> str:
        return _format_index_label(value)

    grouped = []
    for value in sorted(ranked_values):
        grouped.append((value, scatter_df[scatter_df["rank"] == value]))

    for value, subset in grouped:
        if subset.empty:
            continue
        symbols = ["diamond" if flag else "circle" for flag in subset["is_anomaly"]]
        color = rank_colors.get(value, palette[0])
        label = f"Rank {_subset_rank_label(value)}"

        # Build customdata with available fields
        custom_cols: list[str] = ["rank", "task_id", "session_id", "status"]
        # Add role if available
        has_role = "role" in subset.columns and subset["role"].notna().any()
        if has_role:
            custom_cols.append("role")
        # Add optional duration fields if they exist
        for col_name in ["generate_s", "reward_s", "toolcall_s"]:
            if col_name in subset.columns:
                custom_cols.append(col_name)

        customdata = subset[custom_cols].to_numpy()

        # Build hover template dynamically
        hover_parts = [
            "Task %{customdata[1]}, Session %{customdata[2]}",
            "Status: %{customdata[3]}",
        ]
        custom_idx = 4
        if has_role:
            hover_parts.insert(0, "Role: %{customdata[4]}")
            custom_idx += 1
        hover_parts.append("Total duration: %{y:.3f}s")
        if "generate_s" in custom_cols:
            hover_parts.append(f"Generate: %{{customdata[{custom_idx}]:.3f}}s")
            custom_idx += 1
        if "reward_s" in custom_cols:
            hover_parts.append(f"Reward: %{{customdata[{custom_idx}]:.3f}}s")
            custom_idx += 1
        if "toolcall_s" in custom_cols:
            hover_parts.append(f"Tool Call: %{{customdata[{custom_idx}]:.3f}}s")

        hovertemplate = "<br>".join(hover_parts) + "<extra></extra>"

        fig.add_trace(
            go.Scatter(
                x=subset["finalized_ts_offset"],
                y=subset["total_s"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=color,
                    symbol=symbols,
                ),
                hovertemplate=hovertemplate,
                customdata=customdata,
                name=label,
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        legend=dict(
            title="Rank",
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        )
    )
    fig.update_xaxes(title="Timeline (s)", row=row, col=col)
    fig.update_yaxes(title="Total duration (s)", row=row, col=col)
    summary = {
        "p95": p95,
        "p50": p50,
        "x_min": float(scatter_df["finalized_ts_offset"].min()),
        "x_max": float(scatter_df["finalized_ts_offset"].max()),
        "y_min": float(scatter_df["total_s"].min()),
        "y_max": float(scatter_df["total_s"].max()),
    }
    return summary


def _add_overall_histogram(
    fig: go.Figure, df: pd.DataFrame, metric: str, row: int
) -> bool:
    if metric not in df.columns:
        return False
    values = df[metric].dropna()
    if values.empty:
        return False
    meta = HISTOGRAM_METRICS[metric]
    xbins = _compute_histogram_bins(
        values, min_bins=min(120, max(len(values) // 2, 96))
    )
    fig.add_trace(
        go.Histogram(
            x=values,
            name=f"{meta['label']} (overall)",
            marker=dict(color=meta["color"]),
            opacity=0.6,
            bingroup=f"overall_{metric}",
            hovertemplate=(
                f"{meta['label']}<br>Duration: %{{x:.3f}}s" + "<extra></extra>"
            ),
            texttemplate="%{y}",
            textposition="auto",
            showlegend=False,
            autobinx=False,
            xbins=xbins,
        ),
        row=row,
        col=1,
    )
    fig.update_xaxes(title="Duration (s)", row=row, col=1)
    fig.update_yaxes(title="Count", row=row, col=1)
    return True


def _add_step_histograms(
    fig: go.Figure,
    df: pd.DataFrame,
    metric: str,
    row: int,
    step_ids: Sequence[int],
    default_step: int,
) -> dict[int, list[int]]:
    if "step_id" not in df.columns or metric not in df.columns:
        return {}
    step_df = df.dropna(subset=["step_id"])
    if step_df.empty:
        return {}

    meta = HISTOGRAM_METRICS[metric]
    trace_indices: dict[int, list[int]] = {}
    for step in step_ids:
        subset = step_df[step_df["step_id"].astype(int) == step]
        values = subset[metric].dropna()
        min_bins = min(500, max(len(values) * 5, 200))
        xbins = _compute_histogram_bins(
            values,
            max_bins=800,
            min_bins=min_bins,
        )
        hist = go.Histogram(
            x=values,
            name=f"Step {step} · {meta['label']}",
            marker=dict(color=meta["color"]),
            opacity=0.6,
            bingroup=f"step_{metric}",
            hovertemplate=(
                f"Step {step} · {meta['label']}<br>Duration: %{{x:.3f}}s"
                + "<extra></extra>"
            ),
            visible=step == default_step,
            texttemplate="%{y}",
            textposition="auto",
            showlegend=False,
            autobinx=False,
            xbins=xbins,
        )
        fig.add_trace(hist, row=row, col=1)
        trace_indices.setdefault(step, []).append(len(tuple(fig.data)) - 1)

    fig.update_xaxes(title="Duration (s)", row=row, col=1)
    fig.update_yaxes(title="Count (step)", row=row, col=1)
    return trace_indices


def build_overall_distribution_figure(df: pd.DataFrame) -> go.Figure:
    metrics_with_data = [
        metric
        for metric in HISTOGRAM_METRICS
        if metric in df.columns and not df[metric].dropna().empty
    ]

    if not metrics_with_data:
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=("Session distributions",),
        )
        fig.add_annotation(
            text="No distribution metrics available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(
            height=320,
            hovermode="closest",
            template="plotly_white",
            title="Session distribution (overall)",
        )
        return fig

    total_rows = len(metrics_with_data)
    row_heights: list[float] = []
    subplot_titles: list[str] = []
    overall_height = 0.62
    for metric in metrics_with_data:
        meta = HISTOGRAM_METRICS[metric]
        row_heights.append(overall_height)
        subplot_titles.append(f"{meta['label']} (overall)")

    vertical_spacing = 0.12
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        row_heights=row_heights,
        vertical_spacing=vertical_spacing,
        subplot_titles=tuple(subplot_titles),
    )

    current_row = 1
    for metric in metrics_with_data:
        added = _add_overall_histogram(fig, df, metric, current_row)
        if added:
            current_row += 1
    base_height = 420
    per_extra_metric = 220
    height = base_height + per_extra_metric * max(len(metrics_with_data) - 1, 0)
    fig.update_layout(
        height=max(320, height),
        hovermode="closest",
        title="Session distribution (overall)",
        bargap=0.2,
        template="plotly_white",
        margin=dict(t=80, b=80, l=60, r=40),
    )
    return fig


def build_step_distribution_figure(
    df: pd.DataFrame, step_counts: pd.Series | None
) -> go.Figure | None:
    if step_counts is None or step_counts.empty:
        return None

    metrics_with_data = [
        metric
        for metric in HISTOGRAM_METRICS
        if metric in df.columns and not df[metric].dropna().empty
    ]
    if not metrics_with_data:
        return None

    step_ids = sorted(int(step) for step in step_counts.index.tolist())
    default_step = step_ids[0] if step_ids else 0

    total_rows = len(metrics_with_data)
    row_heights = [0.64] * total_rows
    subplot_titles = [
        f"{HISTOGRAM_METRICS[metric]['label']} by step" for metric in metrics_with_data
    ]

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        row_heights=row_heights,
        vertical_spacing=0.16,
        subplot_titles=tuple(subplot_titles),
    )

    current_row = 1
    combined_step_indices: dict[int, list[int]] = {}
    for metric in metrics_with_data:
        indices = _add_step_histograms(
            fig,
            df,
            metric,
            current_row,
            step_ids,
            default_step,
        )
        if indices:
            for step, trace_ids in indices.items():
                combined_step_indices.setdefault(step, []).extend(trace_ids)
        current_row += 1

    if combined_step_indices:
        base_visibility = [
            trace.visible if trace.visible is not None else True for trace in fig.data
        ]
        all_step_traces = {
            idx for indices in combined_step_indices.values() for idx in indices
        }
        buttons: list[dict[str, Any]] = []
        for step in step_ids:
            visibility = base_visibility.copy()
            for idx in all_step_traces:
                visibility[idx] = False
            for idx in combined_step_indices.get(step, []):
                visibility[idx] = True
            buttons.append(
                {
                    "label": f"Step {step}",
                    "method": "update",
                    "args": [
                        {"visible": visibility},
                    ],
                }
            )
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=1.0,
                    y=1.0,
                    xanchor="left",
                    yanchor="top",
                    pad=dict(t=0, b=0, l=10, r=0),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    font=dict(size=12),
                )
            ]
        )

    base_height = 440
    per_extra_metric = 240
    height = base_height + per_extra_metric * max(len(metrics_with_data) - 1, 0)
    fig.update_layout(
        height=max(320, height),
        hovermode="closest",
        title="Session distribution (by step)",
        bargap=0.2,
        template="plotly_white",
        margin=dict(t=80, b=80, l=60, r=180),
    )
    return fig


def build_timeline_figures(
    df: pd.DataFrame, max_sessions: int | None = None
) -> dict[str, go.Figure]:
    """Build timeline visualization showing session lifecycle with phases.

    Args:
        df: DataFrame containing session records
        max_sessions: Optional limit on number of sessions to visualize (earliest finalized first)

    Returns:
        Dictionary mapping rank labels to Plotly figures (e.g., {"r0": fig, "r1": fig, "rNA": fig})
    """
    # Filter to top N sessions by earliest finalized timestamp if limit is specified
    timeline_df = df.dropna(subset=["submit_ts"]).copy()
    total_sessions = len(timeline_df)

    if max_sessions is not None and max_sessions > 0 and total_sessions > max_sessions:
        # Sort by finalized_ts and take the first max_sessions
        timeline_df = timeline_df.sort_values("finalized_ts").head(max_sessions)
        # Re-sort by submit_ts for display
        timeline_df = timeline_df.sort_values("submit_ts")
    else:
        timeline_df = timeline_df.sort_values("submit_ts")

    # Get unique ranks and sort them
    ranked_values = timeline_df["rank"].dropna().unique().tolist()
    include_nan = timeline_df["rank"].isna().any()

    # Raise error if any rank is NA
    if include_nan:
        na_count = timeline_df["rank"].isna().sum()
        raise ValueError(
            f"Found {na_count} session(s) with NA rank. "
            "All sessions must have a valid rank value."
        )

    rank_list = sorted(ranked_values)

    # Calculate base_time once for all ranks using the full timeline_df
    base_time = (
        float(timeline_df["submit_ts"].min())
        if "submit_ts" in timeline_df.columns and not timeline_df.empty
        else 0.0
    )

    # Build separate figures for each rank
    figures: dict[str, go.Figure] = {}

    if not rank_list:
        # No ranks, create a single figure
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=("Session lifecycle",),
        )
        _build_timeline(fig, timeline_df, base_time=base_time, row=1, col=1)
        timeline_count = len(timeline_df)
        height = max(320, min(1200, 180 + timeline_count * 45))
        fig.update_layout(
            height=height,
            hovermode="closest",
            bargap=0.2,
            template="plotly_white",
            title="Session lifecycle",
        )
        figures["all"] = fig
        return figures

    # Create a separate figure for each rank
    for rank_value in rank_list:
        # Filter data for this rank
        rank_df = timeline_df[timeline_df["rank"] == rank_value].copy()
        rank_key = f"r{_format_index_label(rank_value)}"
        rank_label = f"Rank {_format_index_label(rank_value)}"

        if rank_df.empty:
            continue

        # Create a figure for this rank
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=(f"{rank_label} lifecycle",),
        )
        _build_timeline(fig, rank_df, base_time=base_time, row=1, col=1)

        # Calculate height based on number of sessions in this rank
        rank_session_count = len(rank_df)
        height = max(320, min(1200, 180 + rank_session_count * 45))

        fig.update_layout(
            height=height,
            hovermode="closest",
            bargap=0.2,
            template="plotly_white",
            title=f"Session lifecycle - {rank_label}",
            xaxis_title="Time offset (s)",
            yaxis_title="Session",
            margin=dict(t=80, b=80, l=60, r=40),
        )

        figures[rank_key] = fig

    return figures


def build_latency_figure(
    df: pd.DataFrame, step_timepoints: Sequence[float] | None = None
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("Execution duration diagram",),
    )
    summary = _build_latency_scatter(fig, df, row=1, col=1)
    percentile_trace_indices: list[int] = []
    base_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    base_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    step_layout_on: dict[str, Any] | None = None
    step_layout_off = {
        "shapes": deepcopy(base_shapes),
        "annotations": deepcopy(base_annotations),
    }

    if step_timepoints and summary:
        y_min = summary.get("y_min", math.nan)
        y_max = summary.get("y_max", math.nan)
        if not math.isfinite(y_min) or not math.isfinite(y_max):
            y_min, y_max = 0.0, 1.0
        if math.isclose(y_min, y_max):
            delta = max(abs(y_min) * 0.05, 1e-3)
            y_min -= delta
            y_max += delta
        step_shapes: list[dict[str, Any]] = []
        step_annotations: list[dict[str, Any]] = []
        for idx, timepoint in enumerate(step_timepoints, 1):
            if not math.isfinite(timepoint):
                continue
            step_shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="paper",
                    x0=timepoint,
                    x1=timepoint,
                    y0=0.0,
                    y1=1.0,
                    line=dict(color="#888888", dash="dash", width=1.2),
                )
            )
            step_annotations.append(
                dict(
                    x=timepoint,
                    y=1.0,
                    xref="x",
                    yref="paper",
                    text=f"Step {idx}",
                    textangle=-90,
                    xanchor="right",
                    yanchor="top",
                    font=dict(color="#555555", size=11),
                    showarrow=False,
                    xshift=-4,
                    yshift=8,
                )
            )
        if step_shapes or step_annotations:
            combined_shapes = base_shapes + step_shapes
            combined_annotations = base_annotations + step_annotations
            fig.update_layout(
                shapes=combined_shapes,
                annotations=combined_annotations,
            )
            step_layout_on = {
                "shapes": deepcopy(combined_shapes),
                "annotations": deepcopy(combined_annotations),
            }

    if summary:
        x_min = summary.get("x_min", math.nan)
        x_max = summary.get("x_max", math.nan)
        if math.isfinite(x_min) and math.isfinite(x_max):
            if math.isclose(x_min, x_max):
                delta = max(abs(x_min) * 0.05, 0.5)
                x_min -= delta
                x_max += delta
            for value, label, color, dash in (
                (summary.get("p50"), "p50", "#1f77b4", "dot"),
                (summary.get("p95"), "p95", "#d62728", "dash"),
            ):
                if value is None or not math.isfinite(value):
                    continue
                percentile_trace = go.Scatter(
                    x=[x_min, x_max],
                    y=[value, value],
                    mode="lines",
                    line=dict(color=color, dash=dash, width=1.3),
                    name=label,
                    hovertemplate=f"{label}: %{{y:.3f}}s<extra></extra>",
                    showlegend=False,
                    visible=False,
                )
                fig.add_trace(percentile_trace, row=1, col=1)
                percentile_trace_indices.append(len(tuple(fig.data)) - 1)

    toggles: list[dict[str, Any]] = []
    if step_layout_on is not None:
        toggles.append(
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Steps On",
                        method="update",
                        args=[{}, step_layout_on],
                    ),
                    dict(
                        label="Steps Off",
                        method="update",
                        args=[{}, step_layout_off],
                    ),
                ],
                direction="left",
                showactive=True,
                active=0,
                x=0.02,
                y=-0.18,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, r=6),
            )
        )
    if percentile_trace_indices:
        toggles.append(
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Percentiles On",
                        method="restyle",
                        args=[{"visible": True}, percentile_trace_indices],
                    ),
                    dict(
                        label="Percentiles Off",
                        method="restyle",
                        args=[{"visible": False}, percentile_trace_indices],
                    ),
                ],
                direction="left",
                showactive=True,
                active=1,
                x=0.28,
                y=-0.18,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, r=6),
            )
        )
    if toggles:
        fig.update_layout(updatemenus=toggles)
    if not tuple(fig.data):
        fig.add_annotation(
            text="No execution data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
    fig.update_layout(
        height=420,
        hovermode="closest",
        template="plotly_white",
        title="Execution duration diagram",
        margin=dict(t=80, b=90, l=60, r=40),
    )
    return fig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize session-trace JSONL produced by PerfTracer.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path, directory, or glob pattern for PerfTracer session JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional HTML output path (defaults to a path derived from input)",
    )
    parser.add_argument(
        "-B",
        "--consumer-batch-size",
        type=int,
        required=True,
        help="Global consumer batch size (total tasks across all ranks per step, should match controller config)",
    )
    parser.add_argument(
        "-L",
        "--enable-lifecycle",
        action="store_true",
        help="Enable generation of lifecycle timeline figures (one per rank)",
    )
    parser.add_argument(
        "-N",
        "--max-lifecycle-sessions",
        type=int,
        default=None,
        help="Maximum number of sessions to show in lifecycle timeline (earliest finalized first). Only effective when --enable-lifecycle is set.",
    )
    return parser.parse_args(argv)


def _default_output_path(input_arg: str, sources: Sequence[Path]) -> Path:
    if sources:
        if len(sources) == 1:
            base = sources[0]
            return base.with_name(f"{base.stem}-distribution.html")
    candidate = Path(input_arg).expanduser()
    if candidate.exists():
        if candidate.is_dir():
            return candidate / "sessions-distribution.html"
        return candidate.with_name(f"{candidate.stem}-distribution.html")
    stem = candidate.stem or candidate.name or "sessions-distribution"
    safe_stem = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem
    ).strip("_")
    if not safe_stem:
        safe_stem = "sessions-distribution"
    if not safe_stem.endswith("distribution"):
        safe_stem = f"{safe_stem}-distribution"
    parent = (
        candidate.parent
        if candidate.parent not in (Path("."), Path(""))
        else Path.cwd()
    )
    if not parent.exists():
        parent = Path.cwd()
    return parent / f"{safe_stem}.html"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sources = _resolve_record_files(args.input)
    if not sources:
        raise FileNotFoundError(f"No trace files matched input: {args.input}")
    records: list[dict[str, Any]] = []
    for source in sources:
        records.extend(_load_records(source))
    if not records:
        raise ValueError(f"No records found in {args.input}")
    df = pd.DataFrame(records)
    _ensure_numeric(df)
    _maybe_compute_durations(df)
    _extract_phase_timestamps(df)
    _compute_offsets(df)
    step_timepoints, timepoint_method = _determine_step_timepoints(
        df,
        args.consumer_batch_size,
    )
    if step_timepoints:
        method_desc = timepoint_method or "unspecified method"
        print(
            f"Identified {len(step_timepoints)} step timepoint(s) using {method_desc}."
        )
        # Assign steps first to get session counts
        step_counts = _apply_step_assignments(df, step_timepoints)
        for idx, offset in enumerate(step_timepoints, 1):
            session_count = step_counts.get(idx - 1, 0) if not step_counts.empty else 0
            print(f"  Step {idx}: end offset {offset:.3f}s (sessions: {session_count})")
    else:
        reason = timepoint_method or "insufficient data"
        print(f"No step timepoints identified ({reason}).")
        step_counts = _apply_step_assignments(df, step_timepoints)
    if step_counts.empty:
        print("No step assignments available for distribution analysis.")

    overall_fig = build_overall_distribution_figure(df)
    step_fig = build_step_distribution_figure(df, step_counts)

    output_path = args.output or _default_output_path(args.input, sources)
    if args.output:
        overall_path = output_path
        suffix = overall_path.suffix or ".html"
        step_path = overall_path.with_name(f"{overall_path.stem}-step{suffix}")
    else:
        overall_path = output_path.with_name("sessions-distribution-overall.html")
        step_path = output_path.with_name("sessions-distribution-step.html")

    if overall_path.parent != Path(".") and not overall_path.parent.exists():
        overall_path.parent.mkdir(parents=True, exist_ok=True)
    if step_fig is not None and not step_path.parent.exists():
        step_path.parent.mkdir(parents=True, exist_ok=True)

    overall_fig.write_html(overall_path, include_plotlyjs="cdn", full_html=True)
    if step_fig is not None:
        step_fig.write_html(step_path, include_plotlyjs="cdn", full_html=True)

    # Build lifecycle figures with optional session limit (one per rank)
    # Only generate if --enable-lifecycle is set
    if args.enable_lifecycle:
        total_sessions_with_submit = len(df.dropna(subset=["submit_ts"]))
        timeline_figures = build_timeline_figures(df, args.max_lifecycle_sessions)

        # Calculate how many sessions are actually being visualized
        timeline_df = df.dropna(subset=["submit_ts"])
        if (
            args.max_lifecycle_sessions is not None
            and args.max_lifecycle_sessions > 0
            and total_sessions_with_submit > args.max_lifecycle_sessions
        ):
            sessions_visualized = args.max_lifecycle_sessions
            timeline_df = timeline_df.sort_values("finalized_ts").head(
                sessions_visualized
            )
        else:
            sessions_visualized = total_sessions_with_submit

        # Get rank information
        ranked_values = timeline_df["rank"].dropna().unique().tolist()
        include_nan = timeline_df["rank"].isna().any()
        num_ranks = len(ranked_values) + (1 if include_nan else 0)

        if num_ranks > 1:
            print(
                f"Lifecycle visualization: {sessions_visualized} out of "
                f"{total_sessions_with_submit} sessions across {num_ranks} ranks "
                f"(one file per rank)."
            )
        else:
            if (
                args.max_lifecycle_sessions is not None
                and args.max_lifecycle_sessions > 0
                and total_sessions_with_submit > args.max_lifecycle_sessions
            ):
                print(
                    f"Lifecycle visualization limited to {sessions_visualized} out of "
                    f"{total_sessions_with_submit} sessions (earliest finalized first)."
                )
            else:
                print(
                    f"Lifecycle visualization includes all {sessions_visualized} sessions."
                )

        # Write lifecycle figures - one file per rank
        lifecycle_paths: list[Path] = []
        for rank_key, fig in timeline_figures.items():
            lifecycle_path = overall_path.with_name(
                f"sessions-lifecycle-{rank_key}.html"
            )
            if (
                lifecycle_path.parent != Path(".")
                and not lifecycle_path.parent.exists()
            ):
                lifecycle_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(lifecycle_path, include_plotlyjs="cdn", full_html=True)
            lifecycle_paths.append(lifecycle_path)

    latency_fig = build_latency_figure(df, step_timepoints)
    latency_path = overall_path.with_name("sessions-latency.html")
    if latency_path.parent != Path(".") and not latency_path.parent.exists():
        latency_path.parent.mkdir(parents=True, exist_ok=True)
    latency_fig.write_html(latency_path, include_plotlyjs="cdn", full_html=True)

    print(f"Wrote overall distribution report to {overall_path}")
    if step_fig is not None:
        print(f"Wrote step distribution report to {step_path}")
    if args.enable_lifecycle:
        for lifecycle_path in lifecycle_paths:
            print(f"Wrote lifecycle report to {lifecycle_path}")
    print(f"Wrote execution report to {latency_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
