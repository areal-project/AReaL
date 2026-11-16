# SPDX-License-Identifier: Apache-2.0

"""Validation contracts for offline AReaL dataset preparation.

This module is deliberately pure standard library: it never imports torch,
transformers, ``datasets``, or anything else from the training stack. The CLI
wrapper in :mod:`areal.dataset.validate` is responsible for reading records off
disk and handing them over as plain mappings, which keeps the contract logic
cheap to unit-test and safe to run on a CPU-only box.

Two profiles are supported. Both describe *supported subsets* of what AReaL can
consume rather than a universal schema:

``text-rl``
    Records for RL workflows such as ``RLVRWorkflow``. ``messages`` must be a
    non-empty list of chat messages with a recognised role and string content.
    Extra fields are allowed and are forwarded to the reward function, so
    ``--require-field answer`` only checks presence, never the value.

``tokenized-sft``
    Already-tokenised records for ``train_lm``. ``input_ids`` and ``loss_mask``
    must be non-empty, of equal length, and at least one token must be
    supervised.

Diagnostics are deterministic: for a given record the same list of problems is
produced in the same order on every run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

TEXT_RL = "text-rl"
TOKENIZED_SFT = "tokenized-sft"
PROFILES: tuple[str, ...] = (TEXT_RL, TOKENIZED_SFT)

#: Roles accepted inside ``messages``. Anything else is rejected so that a typo
#: such as ``"Role": "User"`` fails loudly instead of silently reaching the
#: chat template.
ALLOWED_ROLES = frozenset({"system", "user", "assistant"})

# Stable error codes. Tests assert on these, so treat them as public API.
CODE_NOT_AN_OBJECT = "not_an_object"
CODE_MISSING_FIELD = "missing_field"
CODE_WRONG_TYPE = "wrong_type"
CODE_EMPTY_LIST = "empty_list"
CODE_UNSUPPORTED_ROLE = "unsupported_role"
CODE_LENGTH_MISMATCH = "length_mismatch"
CODE_NOT_INTEGER = "not_integer"
CODE_NEGATIVE_VALUE = "negative_value"
CODE_NON_BINARY_MASK = "non_binary_mask"
CODE_NO_SUPERVISED_TOKEN = "no_supervised_token"
CODE_EXCEEDS_MAX_LENGTH = "exceeds_max_length"
CODE_MISSING_REQUIRED_FIELD = "missing_required_field"
CODE_INVALID_JSON = "invalid_json"


class _Missing:
    """Sentinel distinguishing an absent key from a key set to ``None``."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


MISSING = _Missing()


@dataclass(frozen=True)
class Diagnostic:
    """One actionable problem attached to a record.

    ``location`` is either ``line <n>`` (1-based, for JSONL input) or
    ``row <n>`` (0-based, for datasets loaded from disk).
    """

    location: str
    field_path: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "location": self.location,
            "field": self.field_path,
            "code": self.code,
            "message": self.message,
        }


@dataclass
class ValidationReport:
    """Aggregate result of scanning a dataset.

    ``checked`` counts every record that carried data (blank JSONL lines are
    ignored). A record with several problems is counted once in ``invalid``.
    ``diagnostics`` holds at most ``max_errors`` entries; ``truncated`` records
    that more were suppressed.
    """

    profile: str
    source: str
    split: str | None = None
    checked: int = 0
    valid: int = 0
    invalid: int = 0
    diagnostics: list[Diagnostic] = field(default_factory=list)
    truncated: bool = False

    @property
    def exit_code(self) -> int:
        """0 when every record is valid, 1 for bad or empty input."""
        if self.checked == 0 or self.invalid > 0:
            return 1
        return 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "source": self.source,
            "split": self.split,
            "checked": self.checked,
            "valid": self.valid,
            "invalid": self.invalid,
            "diagnostics_truncated": self.truncated,
            "diagnostics": [d.as_dict() for d in self.diagnostics],
        }

    def summary_line(self) -> str:
        return f"checked={self.checked} valid={self.valid} invalid={self.invalid}"


def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_messages(record: Mapping[str, Any], location: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    messages = record.get("messages", MISSING)

    if messages is MISSING:
        diagnostics.append(
            Diagnostic(
                location,
                "messages",
                CODE_MISSING_FIELD,
                "missing required field 'messages'",
            )
        )
        return diagnostics

    if not isinstance(messages, (list, tuple)):
        diagnostics.append(
            Diagnostic(
                location,
                "messages",
                CODE_WRONG_TYPE,
                f"'messages' must be a list, got {_type_name(messages)}",
            )
        )
        return diagnostics

    if len(messages) == 0:
        diagnostics.append(
            Diagnostic(
                location,
                "messages",
                CODE_EMPTY_LIST,
                "'messages' must contain at least one message",
            )
        )
        return diagnostics

    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        if not isinstance(message, Mapping):
            diagnostics.append(
                Diagnostic(
                    location,
                    path,
                    CODE_NOT_AN_OBJECT,
                    f"message must be an object, got {_type_name(message)}",
                )
            )
            continue

        role = message.get("role", MISSING)
        if role is MISSING:
            diagnostics.append(
                Diagnostic(
                    location,
                    f"{path}.role",
                    CODE_MISSING_FIELD,
                    "message is missing 'role'",
                )
            )
        elif not isinstance(role, str):
            diagnostics.append(
                Diagnostic(
                    location,
                    f"{path}.role",
                    CODE_WRONG_TYPE,
                    f"'role' must be a string, got {_type_name(role)}",
                )
            )
        elif role not in ALLOWED_ROLES:
            allowed = ", ".join(sorted(ALLOWED_ROLES))
            diagnostics.append(
                Diagnostic(
                    location,
                    f"{path}.role",
                    CODE_UNSUPPORTED_ROLE,
                    f"unsupported role {role!r}; expected one of: {allowed}",
                )
            )

        content = message.get("content", MISSING)
        if content is MISSING:
            diagnostics.append(
                Diagnostic(
                    location,
                    f"{path}.content",
                    CODE_MISSING_FIELD,
                    "message is missing 'content'",
                )
            )
        elif not isinstance(content, str):
            diagnostics.append(
                Diagnostic(
                    location,
                    f"{path}.content",
                    CODE_WRONG_TYPE,
                    f"'content' must be a string, got {_type_name(content)}",
                )
            )

    return diagnostics


def _check_token_sequence(
    record: Mapping[str, Any], name: str, location: str
) -> tuple[list[Any] | None, list[Diagnostic]]:
    """Validate that ``name`` exists, is a non-empty list, and return it."""
    diagnostics: list[Diagnostic] = []
    value = record.get(name, MISSING)

    if value is MISSING:
        diagnostics.append(
            Diagnostic(
                location, name, CODE_MISSING_FIELD, f"missing required field '{name}'"
            )
        )
        return None, diagnostics

    if not isinstance(value, (list, tuple)):
        diagnostics.append(
            Diagnostic(
                location,
                name,
                CODE_WRONG_TYPE,
                f"'{name}' must be a list, got {_type_name(value)}",
            )
        )
        return None, diagnostics

    if len(value) == 0:
        diagnostics.append(
            Diagnostic(location, name, CODE_EMPTY_LIST, f"'{name}' must not be empty")
        )
        return None, diagnostics

    return list(value), diagnostics


def _check_tokenized_sft(
    record: Mapping[str, Any], location: str, max_length: int | None
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []

    input_ids, ids_diag = _check_token_sequence(record, "input_ids", location)
    diagnostics.extend(ids_diag)

    loss_mask, mask_diag = _check_token_sequence(record, "loss_mask", location)
    diagnostics.extend(mask_diag)

    # Element-level checks only make sense once both sequences loaded.
    if input_ids is None or loss_mask is None:
        return diagnostics

    if len(input_ids) != len(loss_mask):
        diagnostics.append(
            Diagnostic(
                location,
                "loss_mask",
                CODE_LENGTH_MISMATCH,
                f"'input_ids' has {len(input_ids)} entries but 'loss_mask' has "
                f"{len(loss_mask)}",
            )
        )

    for index, token in enumerate(input_ids):
        path = f"input_ids[{index}]"
        # bool is a subclass of int, and a boolean token id is never meaningful.
        if isinstance(token, bool) or not isinstance(token, int):
            diagnostics.append(
                Diagnostic(
                    location,
                    path,
                    CODE_NOT_INTEGER,
                    f"token id must be an integer, got {_type_name(token)}",
                )
            )
        elif token < 0:
            diagnostics.append(
                Diagnostic(
                    location,
                    path,
                    CODE_NEGATIVE_VALUE,
                    f"token id must be non-negative, got {token}",
                )
            )

    supervised = 0
    for index, value in enumerate(loss_mask):
        path = f"loss_mask[{index}]"
        if isinstance(value, bool):
            supervised += int(value)
            continue
        if isinstance(value, int) and value in (0, 1):
            supervised += value
            continue
        diagnostics.append(
            Diagnostic(
                location,
                path,
                CODE_NON_BINARY_MASK,
                f"loss mask must be 0, 1, or a boolean, got {value!r}",
            )
        )

    if supervised == 0:
        diagnostics.append(
            Diagnostic(
                location,
                "loss_mask",
                CODE_NO_SUPERVISED_TOKEN,
                "'loss_mask' contains no supervised token",
            )
        )

    if max_length is not None and len(input_ids) > max_length:
        diagnostics.append(
            Diagnostic(
                location,
                "input_ids",
                CODE_EXCEEDS_MAX_LENGTH,
                f"'input_ids' has {len(input_ids)} tokens, exceeds "
                f"--max-length {max_length}",
            )
        )

    return diagnostics


def validate_record(
    record: Any,
    profile: str,
    location: str,
    require_fields: Sequence[str] = (),
    max_length: int | None = None,
) -> list[Diagnostic]:
    """Return every problem found in a single record.

    ``record`` may be any object; non-mappings are rejected up front because
    neither profile can be checked against them.
    """
    if not isinstance(record, Mapping):
        return [
            Diagnostic(
                location,
                "",
                CODE_NOT_AN_OBJECT,
                f"record must be an object, got {_type_name(record)}",
            )
        ]

    if profile == TEXT_RL:
        diagnostics = _check_messages(record, location)
    elif profile == TOKENIZED_SFT:
        diagnostics = _check_tokenized_sft(record, location, max_length)
    else:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")

    for name in require_fields:
        if name not in record:
            diagnostics.append(
                Diagnostic(
                    location,
                    name,
                    CODE_MISSING_REQUIRED_FIELD,
                    f"record is missing required field '{name}'",
                )
            )

    return diagnostics


class ValidationSession:
    """Accumulates a :class:`ValidationReport` across records.

    Scanning always continues to the end of the input; ``max_errors`` only caps
    how many diagnostics are retained, and any suppressed ones raise the report
    ``truncated`` flag.
    """

    def __init__(
        self,
        profile: str,
        source: str,
        split: str | None = None,
        require_fields: Sequence[str] = (),
        max_length: int | None = None,
        max_errors: int | None = None,
    ) -> None:
        self._require_fields = tuple(require_fields)
        self._max_length = max_length
        self._max_errors = max_errors
        self.report = ValidationReport(profile=profile, source=source, split=split)

    def observe(self, record: Any, location: str) -> None:
        diagnostics = validate_record(
            record,
            self.report.profile,
            location,
            require_fields=self._require_fields,
            max_length=self._max_length,
        )
        self.report.checked += 1
        if diagnostics:
            self.report.invalid += 1
        else:
            self.report.valid += 1
        self._absorb(diagnostics)

    def observe_unparseable(self, location: str, reason: str) -> None:
        """Count a record that could not even be decoded as JSON."""
        self.report.checked += 1
        self.report.invalid += 1
        self._absorb(
            [Diagnostic(location, "", CODE_INVALID_JSON, f"invalid JSON: {reason}")]
        )

    def _absorb(self, diagnostics: Iterable[Diagnostic]) -> None:
        for diagnostic in diagnostics:
            if (
                self._max_errors is not None
                and len(self.report.diagnostics) >= self._max_errors
            ):
                self.report.truncated = True
                return
            self.report.diagnostics.append(diagnostic)


def scan_jsonl(path: str, session: ValidationSession) -> None:
    """Validate a JSONL file line by line, streaming and never buffering it.

    Blank lines are skipped and do not count towards ``checked``. A malformed
    line is reported with its 1-based line number and does not stop the scan.
    """
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            location = f"line {line_number}"
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                session.observe_unparseable(location, exc.msg)
                continue
            session.observe(record, location)


def scan_records(records: Iterable[Any], session: ValidationSession) -> None:
    """Validate an iterable of already-decoded records (e.g. a HF Dataset)."""
    for index, record in enumerate(records):
        session.observe(record, f"row {index}")


def format_diagnostics(report: ValidationReport) -> list[str]:
    """Render diagnostics as stable, greppable one-line strings."""
    lines = [
        f"{d.location}: {d.field_path or '<record>'} [{d.code}] {d.message}"
        for d in report.diagnostics
    ]
    if report.truncated:
        lines.append(
            f"... more diagnostics suppressed (showing first {len(report.diagnostics)})"
        )
    return lines


__all__ = [
    "ALLOWED_ROLES",
    "CODE_EMPTY_LIST",
    "CODE_EXCEEDS_MAX_LENGTH",
    "CODE_INVALID_JSON",
    "CODE_LENGTH_MISMATCH",
    "CODE_MISSING_FIELD",
    "CODE_MISSING_REQUIRED_FIELD",
    "CODE_NEGATIVE_VALUE",
    "CODE_NON_BINARY_MASK",
    "CODE_NO_SUPERVISED_TOKEN",
    "CODE_NOT_AN_OBJECT",
    "CODE_NOT_INTEGER",
    "CODE_UNSUPPORTED_ROLE",
    "CODE_WRONG_TYPE",
    "Diagnostic",
    "PROFILES",
    "TEXT_RL",
    "TOKENIZED_SFT",
    "ValidationReport",
    "ValidationSession",
    "format_diagnostics",
    "scan_jsonl",
    "scan_records",
    "validate_record",
]
