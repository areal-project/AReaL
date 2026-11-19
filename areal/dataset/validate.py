# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from areal.dataset.validation import (
    VALID_PROFILES,
    InvalidRecord,
    RecordError,
    RecordLocation,
    ValidationReport,
    validate_records,
)


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate prepared local AReaL data without downloading datasets "
            "or loading models."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Local JSONL file or a Dataset.save_to_disk() directory",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=VALID_PROFILES,
        help="Validation rules to apply",
    )
    parser.add_argument(
        "--split",
        help="DatasetDict split to validate (required for DatasetDict input)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the complete JSON report to this path",
    )
    parser.add_argument(
        "--require-field",
        dest="require_fields",
        action="append",
        default=[],
        metavar="FIELD",
        help=(
            "Require FIELD to exist in each text-rl record; repeat this "
            "option for multiple fields"
        ),
    )
    parser.add_argument(
        "--max-length",
        type=positive_int,
        help="Maximum input_ids length for the tokenized-sft profile",
    )
    parser.add_argument(
        "--max-errors",
        type=non_negative_int,
        default=20,
        help=(
            "Maximum detailed diagnostics to store (default: 20); all "
            "records are still scanned"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        _validate_arguments(parser, args)
    except SystemExit as exc:
        return int(exc.code)

    try:
        return _run(args)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.profile == "tokenized-sft" and args.require_fields:
        parser.error("--require-field is only supported by the text-rl profile")
    if args.profile == "text-rl" and args.max_length is not None:
        parser.error("--max-length is only supported by the tokenized-sft profile")
    if args.split is not None and not args.split.strip():
        parser.error("--split must be a non-empty split name")

    unique_fields: list[str] = []
    for field in args.require_fields:
        if not field.strip():
            parser.error("--require-field must be a non-empty field name")
        if field not in unique_fields:
            unique_fields.append(field)
    args.require_fields = tuple(unique_fields)


def _run(args: argparse.Namespace) -> int:
    input_path = args.path
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    if not input_path.is_file() and not input_path.is_dir():
        raise ValueError(f"Input path is neither a file nor a directory: {input_path}")
    if input_path.is_file() and args.split is not None:
        raise ValueError("--split is only supported for DatasetDict directories")
    if args.report is not None:
        _ensure_report_path_is_safe(input_path, args.report)

    if input_path.is_file():
        records = _iter_jsonl_records(input_path)
    else:
        records = _iter_dataset_records(_load_local_dataset(input_path, args.split))

    report = validate_records(
        records,
        profile=args.profile,
        require_fields=args.require_fields,
        max_length=args.max_length,
        max_diagnostics=args.max_errors,
    )
    if args.report is not None:
        _write_report(report, args.report)

    print(json.dumps(report.to_dict(), indent=2))
    if report.checked == 0:
        print("warning: the selected input contains no records.", file=sys.stderr)
        return 1
    if report.invalid > 0:
        return 1
    return 0


def _iter_jsonl_records(input_path: Path) -> Iterator[tuple[Any, RecordLocation]]:
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            location = RecordLocation(kind="line", number=line_number)
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except (ValueError, RecursionError) as exc:
                if isinstance(exc, json.JSONDecodeError):
                    detail = f"{exc.msg} at column {exc.colno}."
                else:
                    detail = str(exc)
                yield (
                    InvalidRecord(
                        RecordError(
                            field="record",
                            code="invalid_json",
                            message=f"Invalid JSON: {detail}",
                        )
                    ),
                    location,
                )
            else:
                yield record, location


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _load_local_dataset(path: Path, split: str | None) -> Any:
    from datasets import Dataset, DatasetDict, load_from_disk

    try:
        dataset = load_from_disk(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to load local dataset from {path}: {exc}") from exc

    if not isinstance(dataset, (Dataset, DatasetDict)):
        raise ValueError(
            f"Local dataset at {path} has unsupported type {type(dataset).__name__}."
        )

    if isinstance(dataset, DatasetDict):
        available_splits = list(dataset.keys())
        if split is None:
            raise ValueError(
                "DatasetDict input requires --split. Available splits: "
                f"{available_splits}"
            )
        if split not in dataset:
            raise ValueError(
                f"Requested split '{split}' not found. Available splits: "
                f"{available_splits}"
            )
        return dataset[split]

    if split is not None:
        raise ValueError("--split is only supported for DatasetDict directories")
    return dataset


def _iter_dataset_records(dataset: Any) -> Iterator[tuple[Any, RecordLocation]]:
    for row_number, record in enumerate(dataset, start=1):
        yield record, RecordLocation(kind="row", number=row_number)


def _ensure_report_path_is_safe(input_path: Path, report_path: Path) -> None:
    input_resolved = input_path.resolve(strict=False)
    report_resolved = report_path.resolve(strict=False)
    if input_resolved == report_resolved:
        raise ValueError("Report path must not overwrite the input path.")

    if report_path.exists() and input_path.exists():
        if os.path.samefile(input_path, report_path):
            raise ValueError("Report path must not overwrite the input path.")

    if report_path.is_symlink():
        raise ValueError("Report path must not be a symbolic link.")
    if report_path.exists() and report_path.is_dir():
        raise ValueError(f"Report path is a directory: {report_path}")

    if report_path.exists() and report_path.stat().st_nlink > 1:
        raise ValueError("Report path has multiple links and may overwrite input data.")

    if input_path.is_dir():
        try:
            report_resolved.relative_to(input_resolved)
        except ValueError:
            return
        raise ValueError("Report path must be outside the input dataset directory.")


def _write_report(report: ValidationReport, report_path: Path) -> None:
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
