# SPDX-License-Identifier: Apache-2.0

"""Offline dataset validation CLI.

Usage::

    python -m areal.dataset.validate data.jsonl --profile text-rl --require-field answer
    python -m areal.dataset.validate data/ --profile tokenized-sft --split train --max-length 2048

Accepts either a JSONL file or a directory written by ``Dataset.save_to_disk()``.
All contract logic lives in :mod:`areal.dataset.validation`; this module only
parses arguments, reads the input, and renders the report.

Exit codes:

====  ==================================================================
0     every scanned record is valid
1     at least one invalid record, or the input contained no records
2     usage error, input could not be loaded, or the report could not be written
====  ==================================================================

Nothing here downloads data, tokenizes, or touches a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Sequence

from areal.dataset.validation import (
    PROFILES,
    TOKENIZED_SFT,
    ValidationReport,
    ValidationSession,
    format_diagnostics,
    scan_jsonl,
    scan_records,
)

#: Diagnostics kept by default. The scan itself is never truncated.
DEFAULT_MAX_ERRORS = 20

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


class InputError(Exception):
    """Input could not be interpreted; always maps to :data:`EXIT_USAGE`."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m areal.dataset.validate",
        description=(
            "Validate a prepared offline dataset before spending GPU time on it. "
            "Supported profiles describe subsets of the AReaL schemas, not a "
            "universal contract."
        ),
    )
    parser.add_argument(
        "path",
        help="JSONL file, or a directory written by Dataset.save_to_disk()",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=PROFILES,
        help="text-rl checks messages-based records; tokenized-sft checks pre-tokenized ones",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="required when the input is a DatasetDict; ignored otherwise",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="write the full report as JSON to this path",
    )
    parser.add_argument(
        "--require-field",
        action="append",
        default=[],
        metavar="NAME",
        dest="require_field",
        help="also require this field to be present; repeatable (e.g. answer)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        metavar="N",
        help="reject tokenized-sft records longer than N tokens (tokenized-sft only)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        metavar="N",
        help=f"cap how many detailed diagnostics are reported (default: {DEFAULT_MAX_ERRORS})",
    )
    return parser


def _load_dataset_dir(path: str, split: str | None) -> tuple[Iterable, str | None]:
    """Load a ``save_to_disk()`` directory, resolving the split explicitly."""
    from datasets import DatasetDict, load_from_disk

    try:
        loaded = load_from_disk(path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a usage error
        raise InputError(f"cannot load dataset from {path}: {exc}") from exc

    if isinstance(loaded, DatasetDict):
        available = list(loaded.keys())
        if split is None:
            raise InputError(
                f"{path} is a DatasetDict with splits {available}; "
                f"pass --split to choose exactly one"
            )
        if split not in loaded:
            raise InputError(
                f"split {split!r} not found in {path}; available splits: {available}"
            )
        return loaded[split], split

    if split is not None:
        raise InputError(
            f"--split {split!r} was given but {path} holds a single Dataset, "
            f"not a DatasetDict"
        )
    return loaded, None


def _write_report(path: str, report: ValidationReport) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _print_report(report: ValidationReport) -> None:
    print(report.summary_line())
    for line in format_diagnostics(report):
        print(line)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_length is not None:
        if args.profile != TOKENIZED_SFT:
            parser.error("--max-length is only supported with --profile tokenized-sft")
        if args.max_length <= 0:
            parser.error("--max-length must be a positive integer")
    if args.max_errors < 0:
        parser.error("--max-errors must be zero or a positive integer")

    path = args.path
    if not os.path.exists(path):
        print(f"error: no such file or directory: {path}", file=sys.stderr)
        return EXIT_USAGE

    try:
        if os.path.isdir(path):
            records, resolved_split = _load_dataset_dir(path, args.split)
            is_jsonl = False
        elif os.path.isfile(path):
            if args.split is not None:
                raise InputError(
                    f"--split only applies to datasets written by save_to_disk(); "
                    f"{path} is a JSONL file and has no splits"
                )
            records, resolved_split = None, None
            is_jsonl = True
        else:
            raise InputError(f"{path} is neither a regular file nor a directory")
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    session = ValidationSession(
        profile=args.profile,
        source=path,
        split=resolved_split,
        require_fields=args.require_field,
        max_length=args.max_length,
        max_errors=args.max_errors,
    )

    try:
        if is_jsonl:
            scan_jsonl(path, session)
        else:
            scan_records(records, session)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return EXIT_USAGE

    report = session.report
    _print_report(report)

    if args.report is not None:
        try:
            _write_report(args.report, report)
        except OSError as exc:
            print(
                f"error: cannot write report to {args.report}: {exc}", file=sys.stderr
            )
            return EXIT_USAGE

    if report.checked == 0:
        print("error: no records found in the input", file=sys.stderr)

    return report.exit_code


if __name__ == "__main__":
    sys.exit(main())
