"""Helpers for generation JSONL/JSON output handling.

This module centralizes path preparation, resume-from-JSONL, and write
operations used by main_generation. Keeping this logic here keeps
main_generation focused on orchestration and model interaction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd

from verl.utils.hdfs_io import makedirs


def compute_results_dir(model_path: str, output_root: str | os.PathLike[str]) -> Path:
    """Compute the output directory name for a given model checkpoint path.

    Mirrors naming used across generation utilities, handling global_step_* checkpoints.
    """
    output_root_path = Path(os.path.expanduser(str(output_root)))
    model_path_obj = Path(os.path.expanduser(model_path))
    base_name = model_path_obj.name or model_path_obj.parent.name
    if isinstance(base_name, str) and base_name.startswith("global_step_"):
        parent_name = model_path_obj.parent.name or "model"
        base_name = f"{parent_name}_{base_name}"
    return output_root_path / base_name


def infer_data_type(df: pd.DataFrame, dataset_path: str) -> str:
    """Infer dataset name for output filenames.

    Prefers an explicit data_source column, falls back to DataFrame attrs, then dataset path stem.
    """
    if "data_source" in df.columns:
        for value in df["data_source"]:
            if isinstance(value, str) and value.strip():
                return value.strip()

    name = df.attrs.get("dataset_name") or df.attrs.get("name")
    if isinstance(name, str) and name:
        return name

    try:
        return Path(os.path.expanduser(dataset_path)).stem
    except Exception:
        return "dataset"


def prepare_output_paths(
    *,
    model_path: str,
    output_root: str | os.PathLike[str],
    dataset: pd.DataFrame,
    dataset_path: str,
    n_samples: int,
    total_splits: int | None = None,
    current_split: int | None = None,
) -> tuple[Path, Path, Path, str]:
    """Prepare output directory and JSON paths for generation outputs.

    Returns (results_dir, jsonl_path, json_path, data_type).
    """
    results_dir = compute_results_dir(model_path, output_root)
    makedirs(str(results_dir), exist_ok=True)
    data_type = infer_data_type(dataset, dataset_path)
    # Match reference split suffix formatting
    use_split = (total_splits is not None) and (int(total_splits) > 1)
    split_suffix = (
        f"_split{int(current_split)}_of_{int(total_splits)}" if use_split else ""
    )
    jsonl_path = results_dir / f"{data_type}_{n_samples}{split_suffix}.jsonl"
    json_path = results_dir / f"{data_type}_{n_samples}{split_suffix}.json"
    return results_dir, jsonl_path, json_path, data_type


def _read_jsonl_matrix(
    jsonl_path: Path, *, total_messages: int, n_samples: int
) -> tuple[list[list[object | None]], int]:
    """Read an existing JSONL file into a matrix form for resume/completeness checks.

    Returns (responses_by_message, done_count).
    """
    responses_by_message: list[list[object | None]] = [[None] * n_samples for _ in range(total_messages)]
    done_pairs: set[tuple[int, int]] = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            mi = rec.get("message_idx")
            si = rec.get("sample_idx")
            text = rec.get("result")
            if isinstance(mi, int) and isinstance(si, int):
                if 0 <= mi < total_messages and 0 <= si < n_samples:
                    responses_by_message[mi][si] = text
                    done_pairs.add((mi, si))
    return responses_by_message, len(done_pairs)


def check_existing_jsonl_complete(
    *,
    jsonl_path: Path,
    json_path: Path,
    results_dir: Path,
    total_messages: int,
    n_samples: int,
) -> bool:
    """If JSONL exists and is complete, ensure JSON is written and signal early exit.

    Returns True when the caller can skip model loading and exit early.
    """
    if not jsonl_path.exists():
        return False

    try:
        responses_by_message, _ = _read_jsonl_matrix(
            jsonl_path, total_messages=total_messages, n_samples=n_samples
        )
    except Exception as e:
        print(f"Warning: failed to read existing JSONL at {jsonl_path}: {e}")
        return False

    missing = sum(1 for row in responses_by_message for v in row if v is None)
    if missing != 0:
        return False

    # Already complete: write JSON if missing, otherwise warn and exit.
    if json_path.exists():
        print(f"Found complete JSONL at {jsonl_path} and existing JSON at {json_path}; not overwriting.")
        return True

    makedirs(str(results_dir), exist_ok=True)
    try:
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(responses_by_message, handle, ensure_ascii=False)
        print(f"Complete JSONL detected. Wrote aggregated JSON to {json_path} without loading models.")
    except Exception as e:
        print(f"Failed to write JSON to {json_path}: {e}")
    return True


def resume_from_jsonl(
    *, jsonl_path: Path, total_messages: int, n_samples: int
) -> tuple[list[list[object | None]], int]:
    """Load partial results from an existing JSONL, returning the matrix and count.

    Prints a brief status message on success.
    """
    responses_by_message: list[list[object | None]] = [[None] * n_samples for _ in range(total_messages)]
    if not jsonl_path.exists():
        return responses_by_message, 0

    try:
        responses_by_message, count = _read_jsonl_matrix(
            jsonl_path, total_messages=total_messages, n_samples=n_samples
        )
        print(f"Resume: loaded {count} completed samples from existing JSONL at {jsonl_path}")
        return responses_by_message, count
    except Exception as e:
        print(f"Warning: failed to read existing JSONL for resume: {e}")
        return responses_by_message, 0


def open_jsonl_append(jsonl_path: Path):  # -> IO[str]
    """Open the JSONL for appending, creating parent directories as needed."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return open(jsonl_path, "a", encoding="utf-8")


def write_jsonl_record(handle, record: dict) -> None:
    """Write a single JSON object as a line to an open JSONL handle."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json_file(json_path: Path, data: Sequence[Sequence[object]]) -> None:
    """Write list-of-lists JSON to the given path."""
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)


def completion_stats(responses_by_message: Sequence[Sequence[object | None]]) -> tuple[int, int]:
    """Return (missing, total_present) counts for the matrix."""
    missing = sum(1 for row in responses_by_message for v in row if v is None)
    total_present = sum(1 for row in responses_by_message for v in row if v is not None)
    return missing, total_present
