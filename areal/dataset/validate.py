# SPDX-License-Identifier: Apache-2.0

"""Standalone offline validation for locally prepared AReaL datasets.

This module checks *prepared* records against the downstream contracts that
AReaL's generic ``load_from_disk`` fallback and SFT engines rely on, before
any GPU resources are arranged. It supports two bounded profiles:

- ``text-rl``: records for text reinforcement-learning rollouts (for example
  ``RLVRWorkflow``). Requires a non-empty ``messages`` list of message objects
  with roles in ``system``/``user``/``assistant`` and string ``content``.
  Additional fields are permitted; repeatable ``--require-field`` options
  check the presence of reward-specific fields such as ``answer`` without
  scoring their values.
- ``tokenized-sft``: records for tokenized supervised fine-tuning (for example
  ``LMEngine.train_lm``). Requires non-empty, equally sized ``input_ids`` and
  ``loss_mask`` lists, non-negative integer token IDs (booleans excluded),
  binary mask values (integers ``0``/``1`` or booleans), and at least one
  supervised token. ``--max-length`` applies to this profile only.

Both profiles describe *supported subsets* of the AReaL contracts, not a
universal schema; multimodal payloads, tool-call conversations, and DPO pairs
are outside their scope.

Input can be a JSONL file (streamed line by line, so memory use stays bounded
regardless of file size) or a directory saved with Hugging Face
``save_to_disk()``. When the saved object is a ``DatasetDict``, an explicit
``--split`` naming one of its splits is required; otherwise every record of
the requested input is scanned. The input is never modified.

Reports are deterministic: every record is counted, and each invalid record
contributes one or more diagnostics carrying a location (``line N``, 1-based
for JSONL files, or ``row N``, 0-based for dataset rows), a field path, and a
stable error code. Detailed diagnostics are capped by ``--max-errors`` while
the scan continues; the report then flags truncation. The JSON report written
via ``--report`` has the shape::

    {
      "input": "<path as provided>",
      "profile": "text-rl",
      "split": "train" | null,
      "checked": 20, "valid": 0, "invalid": 20,
      "max_errors": 2, "truncated": true,
      "diagnostics": [{"location", "field", "code", "message"}, ...]
    }

Exit codes: ``0`` when every checked record is valid, ``1`` when any record
is invalid or the input contains no records, and ``2`` for usage errors,
input-loading failures, or report-writing failures.

Usage::

    python -m areal.dataset.validate my_data.jsonl --profile text-rl \
        --require-field answer --report report.json

Validation logic is kept separate from argument parsing so it can be reused
directly, e.g. ``validate_record(record, profile="text-rl")``.
"""

import argparse
import json
import os
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

EXIT_VALID = 0
EXIT_INVALID_DATA = 1
EXIT_USAGE_ERROR = 2

TEXT_RL_PROFILE = "text-rl"
TOKENIZED_SFT_PROFILE = "tokenized-sft"
PROFILES = (TEXT_RL_PROFILE, TOKENIZED_SFT_PROFILE)

ALLOWED_MESSAGE_ROLES = ("system", "user", "assistant")
DEFAULT_MAX_ERRORS = 10


class DatasetValidationUsageError(ValueError):
    """Raised for invalid options, missing/unknown splits, or bad paths."""


class DatasetInputError(Exception):
    """Raised when the input cannot be loaded at all."""


@dataclass(frozen=True)
class Diagnostic:
    """One deterministic finding for a single field of a single record."""

    location: str
    field_path: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "location": self.location,
            "field": self.field_path,
            "code": self.code,
            "message": self.message,
        }

    def __str__(self) -> str:
        return f"{self.location} {self.field_path} {self.code}: {self.message}"


def _validate_text_rl_record(
    record: Any,
    *,
    require_fields: tuple[str, ...] = (),
    max_length: int | None = None,
) -> list[Diagnostic]:
    """Check one record against the ``text-rl`` contract (prompt + metadata).

    ``max_length`` is accepted but ignored for symmetry with the dispatchers;
    the CLI rejects it for this profile.
    """
    if not isinstance(record, Mapping):
        return [
            Diagnostic(
                location="",
                field_path="record",
                code="record_not_object",
                message=f"record must be a JSON object, got {type(record).__name__}",
            )
        ]

    diagnostics: list[Diagnostic] = []
    messages = record.get("messages")
    if messages is None:
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="messages",
                code="messages_missing",
                message="record has no 'messages' field",
            )
        )
    elif not isinstance(messages, list):
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="messages",
                code="messages_not_list",
                message=f"'messages' must be a list, got {type(messages).__name__}",
            )
        )
    elif not messages:
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="messages",
                code="messages_empty",
                message="'messages' must contain at least one message",
            )
        )
    else:
        for idx, message in enumerate(messages):
            base = f"messages[{idx}]"
            if not isinstance(message, Mapping):
                diagnostics.append(
                    Diagnostic(
                        location="",
                        field_path=base,
                        code="message_not_object",
                        message=(
                            f"{base} must be an object with 'role' and "
                            f"'content', got {type(message).__name__}"
                        ),
                    )
                )
                continue
            if "role" not in message:
                diagnostics.append(
                    Diagnostic(
                        location="",
                        field_path=f"{base}.role",
                        code="role_missing",
                        message=f"{base} has no 'role' field",
                    )
                )
            else:
                role = message["role"]
                if not isinstance(role, str) or role not in ALLOWED_MESSAGE_ROLES:
                    diagnostics.append(
                        Diagnostic(
                            location="",
                            field_path=f"{base}.role",
                            code="role_invalid",
                            message=(
                                f"{base}.role must be one of "
                                f"{list(ALLOWED_MESSAGE_ROLES)}, got {role!r}"
                            ),
                        )
                    )
            if "content" not in message:
                diagnostics.append(
                    Diagnostic(
                        location="",
                        field_path=f"{base}.content",
                        code="content_missing",
                        message=f"{base} has no 'content' field",
                    )
                )
            elif not isinstance(message["content"], str):
                diagnostics.append(
                    Diagnostic(
                        location="",
                        field_path=f"{base}.content",
                        code="content_not_string",
                        message=(
                            f"{base}.content must be a string, got "
                            f"{type(message['content']).__name__}"
                        ),
                    )
                )

    for name in require_fields:
        if name not in record:
            diagnostics.append(
                Diagnostic(
                    location="",
                    field_path=name,
                    code="required_field_missing",
                    message=f"required field {name!r} is missing",
                )
            )
    return diagnostics


def _validate_tokenized_sft_record(
    record: Any,
    *,
    require_fields: tuple[str, ...] = (),
    max_length: int | None = None,
) -> list[Diagnostic]:
    """Check one record against the ``tokenized-sft`` token/mask contract."""
    if not isinstance(record, Mapping):
        return [
            Diagnostic(
                location="",
                field_path="record",
                code="record_not_object",
                message=f"record must be a JSON object, got {type(record).__name__}",
            )
        ]

    diagnostics: list[Diagnostic] = []

    input_ids = record.get("input_ids")
    if input_ids is None:
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="input_ids",
                code="input_ids_missing",
                message="record has no 'input_ids' field",
            )
        )
    elif not isinstance(input_ids, list):
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="input_ids",
                code="input_ids_not_list",
                message=(f"'input_ids' must be a list, got {type(input_ids).__name__}"),
            )
        )
    elif not input_ids:
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="input_ids",
                code="input_ids_empty",
                message="'input_ids' must contain at least one token id",
            )
        )
    else:
        if max_length is not None and len(input_ids) > max_length:
            diagnostics.append(
                Diagnostic(
                    location="",
                    field_path="input_ids",
                    code="exceeds_max_length",
                    message=(
                        f"'input_ids' has {len(input_ids)} tokens, longer than "
                        f"--max-length {max_length}"
                    ),
                )
            )
        for idx, token_id in enumerate(input_ids):
            base = f"input_ids[{idx}]"
            if isinstance(token_id, bool):
                diagnostics.append(
                    Diagnostic(
                        location="",
                        field_path=base,
                        code="token_id_bool",
                        message=f"{base} is a boolean; token ids must be integers",
                    )
                )
            elif not isinstance(token_id, int):
                diagnostics.append(
                    Diagnostic(
                        location="",
                        field_path=base,
                        code="token_id_not_int",
                        message=(
                            f"{base} must be an integer, got {type(token_id).__name__}"
                        ),
                    )
                )
            elif token_id < 0:
                diagnostics.append(
                    Diagnostic(
                        location="",
                        field_path=base,
                        code="token_id_negative",
                        message=f"{base} must be non-negative, got {token_id}",
                    )
                )

    loss_mask = record.get("loss_mask")
    if loss_mask is None:
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="loss_mask",
                code="loss_mask_missing",
                message="record has no 'loss_mask' field",
            )
        )
    elif not isinstance(loss_mask, list):
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="loss_mask",
                code="loss_mask_not_list",
                message=(f"'loss_mask' must be a list, got {type(loss_mask).__name__}"),
            )
        )
    elif not loss_mask:
        diagnostics.append(
            Diagnostic(
                location="",
                field_path="loss_mask",
                code="loss_mask_empty",
                message="'loss_mask' must contain at least one value",
            )
        )
    else:
        if isinstance(input_ids, list) and len(input_ids) != len(loss_mask):
            diagnostics.append(
                Diagnostic(
                    location="",
                    field_path="loss_mask",
                    code="length_mismatch",
                    message=(
                        f"'loss_mask' has {len(loss_mask)} values but "
                        f"'input_ids' has {len(input_ids)}"
                    ),
                )
            )
        supervised = 0
        for idx, mask_value in enumerate(loss_mask):
            if isinstance(mask_value, int) and mask_value in (0, 1):
                if mask_value:
                    supervised += 1
            else:
                diagnostics.append(
                    Diagnostic(
                        location="",
                        field_path=f"loss_mask[{idx}]",
                        code="mask_not_binary",
                        message=(
                            f"loss_mask[{idx}] must be 0/1 or a boolean, "
                            f"got {mask_value!r}"
                        ),
                    )
                )
        if supervised == 0:
            diagnostics.append(
                Diagnostic(
                    location="",
                    field_path="loss_mask",
                    code="no_supervised_tokens",
                    message=(
                        "'loss_mask' has no supervised token (all values are "
                        "0 or False)"
                    ),
                )
            )

    for name in require_fields:
        if name not in record:
            diagnostics.append(
                Diagnostic(
                    location="",
                    field_path=name,
                    code="required_field_missing",
                    message=f"required field {name!r} is missing",
                )
            )
    return diagnostics


_RECORD_VALIDATORS = {
    TEXT_RL_PROFILE: _validate_text_rl_record,
    TOKENIZED_SFT_PROFILE: _validate_tokenized_sft_record,
}


def validate_record(
    record: Any,
    profile: str = TEXT_RL_PROFILE,
    *,
    require_fields: tuple[str, ...] = (),
    max_length: int | None = None,
) -> list[Diagnostic]:
    """Validate one prepared record and return its diagnostics.

    The returned list is empty when the record satisfies the profile. An
    unknown ``profile`` raises :class:`DatasetValidationUsageError`.
    """
    try:
        validator = _RECORD_VALIDATORS[profile]
    except KeyError as exc:
        raise DatasetValidationUsageError(
            f"unknown profile {profile!r}; supported profiles: {', '.join(PROFILES)}"
        ) from exc
    return validator(record, require_fields=require_fields, max_length=max_length)


@dataclass
class ScanResult:
    """Mutable counters accumulated while scanning all records."""

    checked: int = 0
    valid: int = 0
    invalid: int = 0
    diagnostics: list[Diagnostic] = field(default_factory=list)
    truncated: bool = False


def _record_diagnostic(
    result: ScanResult, diagnostic: Diagnostic, location: str, max_errors: int
) -> None:
    if len(result.diagnostics) < max_errors:
        result.diagnostics.append(
            Diagnostic(
                location=location,
                field_path=diagnostic.field_path,
                code=diagnostic.code,
                message=diagnostic.message,
            )
        )
    else:
        result.truncated = True


def scan_records(
    items: Iterator[tuple[str, Any, Diagnostic | None]],
    profile: str,
    *,
    require_fields: tuple[str, ...] = (),
    max_length: int | None = None,
    max_errors: int = DEFAULT_MAX_ERRORS,
) -> ScanResult:
    """Scan ``(location, record, parse_diagnostic)`` triples for one profile.

    Every item is checked regardless of earlier failures; at most
    ``max_errors`` detailed diagnostics are retained, and the result flags
    truncation when further diagnostics were dropped.
    """
    result = ScanResult()
    for location, record, parse_diagnostic in items:
        result.checked += 1
        if parse_diagnostic is not None:
            result.invalid += 1
            _record_diagnostic(result, parse_diagnostic, location, max_errors)
            continue
        diagnostics = validate_record(
            record, profile, require_fields=require_fields, max_length=max_length
        )
        if not diagnostics:
            result.valid += 1
            continue
        result.invalid += 1
        for diagnostic in diagnostics:
            _record_diagnostic(result, diagnostic, location, max_errors)
    return result


@dataclass(frozen=True)
class ValidationReport:
    """Deterministic summary of one scan, ready for JSON serialization."""

    input_path: str
    profile: str
    split: str | None
    checked: int
    valid: int
    invalid: int
    max_errors: int
    truncated: bool
    diagnostics: tuple[Diagnostic, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input_path,
            "profile": self.profile,
            "split": self.split,
            "checked": self.checked,
            "valid": self.valid,
            "invalid": self.invalid,
            "max_errors": self.max_errors,
            "truncated": self.truncated,
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n"


def _iter_jsonl_records(
    path: str,
) -> Iterator[tuple[str, Any, Diagnostic | None]]:
    """Stream a JSONL file, yielding one item per non-blank line.

    Files are read line by line so memory use does not grow with file size.
    A blank line is skipped entirely; a line that is not valid JSON counts
    as one invalid record with a ``json_invalid_line`` diagnostic.
    """
    with open(path, encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            location = f"line {line_no}"
            if not line.strip():
                continue
            try:
                yield location, json.loads(line), None
            except json.JSONDecodeError as exc:
                diagnostic = Diagnostic(
                    location=location,
                    field_path="record",
                    code="json_invalid_line",
                    message=f"invalid JSON: {exc.msg} (column {exc.colno})",
                )
                yield location, None, diagnostic


def _iter_dataset_records(
    dataset: Any,
) -> Iterator[tuple[str, Any, Diagnostic | None]]:
    """Yield one item per row of a Hugging Face ``Dataset`` (0-based rows)."""
    for idx, record in enumerate(dataset):
        yield f"row {idx}", record, None


def _load_saved_dataset(path: str, split: str | None):
    """Load a ``save_to_disk()`` directory and select the records to scan.

    Returns ``(items, split_used)``. ``DatasetDict`` inputs require an
    explicit, existing ``--split``.
    """
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise DatasetInputError(
            "loading a saved dataset directory requires the 'datasets' "
            "package; install it or point at a JSONL file instead"
        ) from exc

    try:
        dataset = load_from_disk(path)
    except Exception as exc:
        raise DatasetInputError(
            f"failed to load a Hugging Face dataset from {path!r}: {exc}"
        ) from exc

    if isinstance(dataset, DatasetDict):
        available = list(dataset.keys())
        if split is None:
            raise DatasetValidationUsageError(
                f"{path!r} is a DatasetDict; pass --split with one of the "
                f"available splits: {available}"
            )
        if split not in dataset:
            raise DatasetValidationUsageError(
                f"requested split {split!r} not found in DatasetDict at "
                f"{path!r}; available splits: {available}"
            )
        return _iter_dataset_records(dataset[split]), split

    if split is not None:
        raise DatasetValidationUsageError(
            f"--split is only applicable to DatasetDict inputs; the data at "
            f"{path!r} is a single dataset"
        )
    return _iter_dataset_records(dataset), None


def _open_input(path: str, split: str | None):
    """Resolve ``path`` into an iterable of ``(location, record, diag)``."""
    if os.path.isdir(path):
        return _load_saved_dataset(path, split)
    if os.path.isfile(path):
        if split is not None:
            raise DatasetValidationUsageError(
                "--split is only applicable to DatasetDict inputs, not JSONL files"
            )
        return _iter_jsonl_records(path), None
    raise DatasetValidationUsageError(f"input path does not exist: {path!r}")


def _check_options(args: argparse.Namespace) -> None:
    """Turn parsed options into validated scan parameters (usage errors)."""
    if args.max_errors < 1:
        raise DatasetValidationUsageError("--max-errors must be a positive integer")
    for name in args.require_fields:
        if not name:
            raise DatasetValidationUsageError(
                "--require-field names must be non-empty strings"
            )
    if args.max_length is not None:
        if args.max_length < 1:
            raise DatasetValidationUsageError("--max-length must be a positive integer")
        if args.profile != TOKENIZED_SFT_PROFILE:
            raise DatasetValidationUsageError(
                f"--max-length is only supported for the "
                f"{TOKENIZED_SFT_PROFILE!r} profile, not {args.profile!r}"
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser (kept separate from validation logic)."""
    parser = argparse.ArgumentParser(
        prog="python -m areal.dataset.validate",
        description=(
            "Validate locally prepared AReaL datasets on CPU before "
            "arranging GPU resources. Accepts a JSONL file or a directory "
            "saved with Hugging Face save_to_disk()."
        ),
        epilog=(
            "Profiles describe supported subsets: 'text-rl' for message "
            "prompts used by text RL workflows, and 'tokenized-sft' for "
            "input_ids/loss_mask records consumed by SFT training. Exit "
            "codes: 0 all records valid, 1 invalid or empty input, "
            "2 usage, loading, or report-writing failure."
        ),
    )
    parser.add_argument(
        "path",
        help="JSONL file, or directory saved with datasets.save_to_disk()",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=TEXT_RL_PROFILE,
        help=f"Contract to validate against (default: {TEXT_RL_PROFILE})",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=("DatasetDict split to scan; required when the input is a DatasetDict"),
    )
    parser.add_argument(
        "--require-field",
        dest="require_fields",
        action="append",
        default=[],
        metavar="FIELD",
        help=(
            "Extra field that must be present in every record (repeatable), "
            "e.g. a reward-specific 'answer'. Presence only; values are not "
            "scored."
        ),
    )
    parser.add_argument(
        "--max-length",
        dest="max_length",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Fail records whose 'input_ids' is longer than N tokens "
            f"(positive integer, {TOKENIZED_SFT_PROFILE} profile only)"
        ),
    )
    parser.add_argument(
        "--max-errors",
        dest="max_errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        metavar="N",
        help=(
            f"Maximum number of detailed diagnostics to retain (positive "
            f"integer, default: {DEFAULT_MAX_ERRORS}); the scan still counts "
            "every record and the report flags truncation"
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="Write the full JSON validation report to PATH",
    )
    return parser


def _print_summary(report: ValidationReport, stream=None) -> None:
    out = sys.stdout if stream is None else stream
    line = (
        f"profile={report.profile} input={report.input_path} "
        + (f"split={report.split} " if report.split is not None else "")
        + f"checked={report.checked} valid={report.valid} "
        f"invalid={report.invalid}"
    )
    print(line, file=out)
    for diagnostic in report.diagnostics:
        print(f"  {diagnostic}", file=out)
    if report.truncated:
        print(
            f"  ... further diagnostics omitted (--max-errors="
            f"{report.max_errors}); see the report for truncation status",
            file=out,
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m areal.dataset.validate``.

    Returns the process exit code: 0 valid, 1 invalid records or empty
    input, 2 usage/loading/report errors.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _check_options(args)
        items, split_used = _open_input(args.path, args.split)
        result = scan_records(
            items,
            args.profile,
            require_fields=tuple(args.require_fields),
            max_length=args.max_length,
            max_errors=args.max_errors,
        )
    except DatasetValidationUsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except (DatasetInputError, OSError, UnicodeDecodeError) as exc:
        print(f"error: failed to read input: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR

    report = ValidationReport(
        input_path=args.path,
        profile=args.profile,
        split=split_used,
        checked=result.checked,
        valid=result.valid,
        invalid=result.invalid,
        max_errors=args.max_errors,
        truncated=result.truncated,
        diagnostics=tuple(result.diagnostics),
    )
    _print_summary(report)

    if args.report is not None:
        try:
            with open(args.report, "w", encoding="utf-8") as fout:
                fout.write(report.to_json())
        except OSError as exc:
            print(
                f"error: failed to write report to {args.report!r}: {exc}",
                file=sys.stderr,
            )
            return EXIT_USAGE_ERROR

    if result.checked == 0 or result.invalid > 0:
        return EXIT_INVALID_DATA
    return EXIT_VALID


if __name__ == "__main__":
    sys.exit(main())
