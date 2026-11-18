# SPDX-License-Identifier: Apache-2.0

"""Check local, prepared AReaL datasets on CPU."""

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

Profile = Literal["text-rl", "tokenized-sft"]


@dataclass(frozen=True)
class Problem:
    field: str
    code: str


def validate_record(
    record: object,
    profile: Profile,
    required_fields: Iterable[str] = (),
    max_length: int | None = None,
) -> Iterator[Problem]:
    """Yield field problems in input order for one prepared record."""
    if not isinstance(record, dict):
        yield Problem("$", "expected_object")
        return

    if profile == "text-rl":
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            yield Problem("messages", "expected_nonempty_list")
        else:
            for index, message in enumerate(messages):
                path = f"messages[{index}]"
                if not isinstance(message, dict):
                    yield Problem(path, "expected_object")
                    continue
                if message.get("role") not in ("system", "user", "assistant"):
                    yield Problem(f"{path}.role", "invalid_role")
                if not isinstance(message.get("content"), str):
                    yield Problem(f"{path}.content", "expected_string")
        for name in required_fields:
            if name not in record:
                yield Problem(name, "missing_field")
        return

    if profile != "tokenized-sft":
        raise ValueError(f"Unknown profile {profile!r}")
    ids, mask = record.get("input_ids"), record.get("loss_mask")
    if not isinstance(ids, list) or not ids:
        yield Problem("input_ids", "expected_nonempty_list")
    else:
        if max_length is not None and len(ids) > max_length:
            yield Problem("input_ids", "over_max_length")
        for index, token in enumerate(ids):
            if type(token) is not int or token < 0:
                yield Problem(f"input_ids[{index}]", "invalid_token_id")

    if not isinstance(mask, list) or not mask:
        yield Problem("loss_mask", "expected_nonempty_list")
    else:
        if isinstance(ids, list) and len(ids) != len(mask):
            yield Problem("loss_mask", "length_mismatch")
        supervised = False
        for index, value in enumerate(mask):
            if not isinstance(value, int) or value not in (0, 1):
                yield Problem(f"loss_mask[{index}]", "invalid_mask_value")
            elif value:
                supervised = True
        if not supervised:
            yield Problem("loss_mask", "no_supervised_token")


@dataclass
class Report:
    checked: int = 0
    valid: int = 0
    invalid: int = 0
    diagnostics: list[dict[str, str | int]] = field(default_factory=list)
    truncated: bool = False


def _invalid_constant(value: str) -> None:
    raise ValueError(f"{value} is not a JSON value")


def _records(
    path: Path, split: str | None
) -> Iterator[tuple[str, int, object, Problem | None]]:
    if path.is_dir():
        from datasets import DatasetDict, load_from_disk

        try:
            dataset = load_from_disk(str(path))
        except Exception as exc:
            raise ValueError(f"Cannot load dataset at {path}: {exc}") from exc
        if isinstance(dataset, DatasetDict):
            if split not in dataset:
                raise ValueError(f"Select --split from {list(dataset)} (got {split!r})")
            dataset = dataset[split]
        elif split is not None:
            raise ValueError("--split requires a DatasetDict")
        for row, record in enumerate(dataset):
            yield "row", row, record, None
        return

    if split is not None:
        raise ValueError("--split requires a DatasetDict")
    with path.open(encoding="utf-8") as source:
        for line, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw, parse_constant=_invalid_constant)
            except ValueError:
                yield "line", line, None, Problem("$", "invalid_json")
            else:
                yield "line", line, record, None


def scan(
    path: Path,
    profile: Profile,
    split: str | None,
    required_fields: list[str],
    max_length: int | None,
    max_errors: int,
) -> Report:
    report = Report()
    for unit, index, record, parse_error in _records(path, split):
        report.checked += 1
        problems = (
            (parse_error,)
            if parse_error
            else validate_record(record, profile, required_fields, max_length)
        )
        invalid = False
        for problem in problems:
            invalid = True
            if len(report.diagnostics) == max_errors:
                report.truncated = True
                break
            report.diagnostics.append({unit: index, **asdict(problem)})
        if invalid:
            report.invalid += 1
        else:
            report.valid += 1
    return report


def _write_report(target: Path, source: Path, report: Report) -> None:
    # Replace the report entry only after writing it. Other hard links keep their data.
    destination = target.parent.resolve() / target.name
    if destination.is_relative_to(source.resolve()) or (
        target.exists() and source.samefile(target)
    ):
        raise ValueError(f"Choose a report path outside the input: {target}")
    output = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, delete=False
    )
    temporary = Path(output.name)
    try:
        with output:
            json.dump(asdict(report), output, indent=2)
            output.write("\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path, help="local JSONL file or save_to_disk directory"
    )
    parser.add_argument(
        "--profile", required=True, choices=("text-rl", "tokenized-sft")
    )
    parser.add_argument("--split", help="select a DatasetDict split")
    parser.add_argument(
        "--require-field",
        action="append",
        default=[],
        help="required text-rl field (repeatable)",
    )
    parser.add_argument(
        "--max-length", type=int, help="maximum tokenized-sft sequence length"
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="maximum retained diagnostics (default: 20)",
    )
    parser.add_argument("--report", type=Path, help="write a JSON report")
    args = parser.parse_args(argv)
    if args.max_errors < 0:
        parser.error("--max-errors must be non-negative")
    if args.max_length is not None and (
        args.profile != "tokenized-sft" or args.max_length <= 0
    ):
        parser.error("--max-length requires tokenized-sft and a positive length")
    if args.require_field and args.profile != "text-rl":
        parser.error("--require-field requires text-rl")
    try:
        report = scan(
            args.path,
            args.profile,
            args.split,
            args.require_field,
            args.max_length,
            args.max_errors,
        )
        if args.report is not None:
            _write_report(args.report, args.path, report)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    print(f"checked={report.checked} valid={report.valid} invalid={report.invalid}")
    for diagnostic in report.diagnostics:
        print(json.dumps(diagnostic, ensure_ascii=False))
    if report.truncated:
        print("Further diagnostics truncated.")
    return 0 if report.checked and not report.invalid else 1


if __name__ == "__main__":
    sys.exit(main())
