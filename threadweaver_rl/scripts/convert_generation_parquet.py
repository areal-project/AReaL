#!/home/longlian/miniconda3/envs/verl/bin/python
"""Convert generation parquet outputs into JSONL + JSON bundles.

The vLLM async branching generation writes responses back into the input parquet
under a ``responses`` column.  This utility mirrors the naming convention of
``eval_aime2024_api_multiverse_ds-qwen-general_v1.py`` by producing a directory
named after the model checkpoint and saving both a resumable JSONL log and the
final JSON list-of-lists bundle.

Example::

    python scripts/convert_generation_parquet.py \
        /tmp/parallel_outputs.parquet \
        --model-path ~/Multiverse/ckpts/32n-p1-40k-pt225613-4bd-pfrv2.3npfe_a0.1af15am0.2_rms-0914_194012_global_step_400_hf \
        --output-root ./outputs \
        --data-type aime
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - surfaced immediately when missing
    raise SystemExit(
        "pandas is required to convert the generation parquet; please install it in the current environment"
    ) from exc


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert generation parquet to JSON outputs")
    parser.add_argument("input_parquet", type=Path, help="Path to the parquet file produced by main_generation")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model checkpoint path used for generation (drives the output directory name)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.cwd(),
        help="Base directory where the model-specific folder will be created (default: current directory)",
    )
    parser.add_argument(
        "--data-type",
        default=None,
        help="Dataset name to use in the output filenames; inferred when omitted.",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Optional suffix appended to the model directory name (e.g. 'bfloat16_wait600').",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=None,
        help="Optional dataset split index (0-based) to include in the filenames.",
    )
    parser.add_argument(
        "--split-total",
        type=int,
        default=None,
        help="Optional number of total splits to include in the filenames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing JSON/JSONL files to be replaced.",
    )
    return parser.parse_args(argv)


def _normalize_responses_column(responses: Iterable) -> List[str]:
    """Convert the per-row responses payload to a plain list of strings."""

    if responses is None:
        return []

    if isinstance(responses, (list, tuple)):
        normalized = list(responses)
    else:
        try:
            normalized = list(responses)
        except TypeError:
            normalized = [responses]

    return ["" if r is None else str(r) for r in normalized]


def _compute_results_dir(model_path: str, output_root: Path, suffix: str | None) -> Path:
    model_path_obj = Path(model_path)
    base_name = model_path_obj.name or model_path_obj.parent.name
    if base_name.startswith("global_step_"):
        parent_name = model_path_obj.parent.name or "model"
        base_name = f"{parent_name}_{base_name}"
    if suffix:
        base_name = f"{base_name}_{suffix.strip()}"
    return output_root / base_name


def _infer_data_type(df: pd.DataFrame, fallback: str | None) -> str:
    if fallback:
        return fallback

    if "data_source" in df.columns:
        for value in df["data_source"]:
            if isinstance(value, str) and value.strip():
                return value.strip()

    return df.attrs.get("dataset_name") or df.attrs.get("name") or "dataset"


def _build_split_suffix(index: int | None, total: int | None) -> str:
    if index is None and total is None:
        return ""
    if index is None or total is None:
        raise ValueError("split-index and split-total must be provided together")
    if index < 0 or total <= 0 or index >= total:
        raise ValueError(f"Invalid split settings: index={index}, total={total}")
    return f"_split{index}_of_{total}"


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if not args.input_parquet.exists():
        raise FileNotFoundError(f"Input parquet file not found: {args.input_parquet}")

    df = pd.read_parquet(args.input_parquet)
    if "responses" not in df.columns:
        raise KeyError("Expected 'responses' column in the parquet output")
    if df.empty:
        raise ValueError("The parquet file does not contain any rows to convert")

    responses_per_prompt: List[List[str]] = []
    n_samples: int | None = None
    for row_idx, payload in enumerate(df["responses"]):
        normalized = _normalize_responses_column(payload)
        if n_samples is None:
            n_samples = len(normalized)
            if n_samples == 0:
                raise ValueError("Responses column is empty for the first row")
        elif len(normalized) != n_samples:
            raise ValueError(
                f"Row {row_idx} has {len(normalized)} samples, expected {n_samples}"
            )
        responses_per_prompt.append(normalized)

    assert n_samples is not None  # for type checkers

    data_type = _infer_data_type(df, args.data_type)
    split_suffix = _build_split_suffix(args.split_index, args.split_total)

    results_dir = _compute_results_dir(args.model_path, args.output_root, args.suffix)
    results_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = results_dir / f"{data_type}_{n_samples}{split_suffix}.jsonl"
    json_path = results_dir / f"{data_type}_{n_samples}{split_suffix}.json"

    if not args.overwrite:
        for path in (jsonl_path, json_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    total_entries = len(responses_per_prompt) * n_samples
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for message_idx, samples in enumerate(responses_per_prompt):
            for sample_idx, text in enumerate(samples):
                record = {"message_idx": message_idx, "sample_idx": sample_idx, "result": text}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(responses_per_prompt, handle, ensure_ascii=False)

    print(f"Wrote {total_entries} samples for {len(responses_per_prompt)} prompts")
    print(f"JSONL: {jsonl_path}")
    print(f"JSON:  {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
