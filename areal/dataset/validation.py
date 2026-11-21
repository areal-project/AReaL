# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

Profile = Literal["text-rl", "tokenized-sft"]
VALID_PROFILES: tuple[Profile, ...] = ("text-rl", "tokenized-sft")
VALID_ROLES: tuple[str, ...] = ("system", "user", "assistant")


@dataclass(frozen=True)
class RecordLocation:
    kind: Literal["line", "row"]
    number: int

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "number": self.number}


@dataclass(frozen=True)
class RecordError:
    field: str
    code: str
    message: str


@dataclass(frozen=True)
class InvalidRecord:
    error: RecordError


@dataclass(frozen=True)
class Diagnostic:
    location: RecordLocation
    field: str
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location.to_dict(),
            "field": self.field,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationReport:
    profile: Profile
    checked: int
    valid: int
    invalid: int
    diagnostics: tuple[Diagnostic, ...]
    diagnostics_truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "checked": self.checked,
            "valid": self.valid,
            "invalid": self.invalid,
            "diagnostics_truncated": self.diagnostics_truncated,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


def iter_record_errors(
    record: Any,
    profile: Profile,
    *,
    require_fields: Iterable[str] = (),
    max_length: int | None = None,
) -> Iterator[RecordError]:
    require_fields = tuple(require_fields)
    if profile not in VALID_PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    if profile == "text-rl" and max_length is not None:
        raise ValueError("max_length is only supported by the tokenized-sft profile")
    if profile == "tokenized-sft" and require_fields:
        raise ValueError("require_fields is only supported by the text-rl profile")
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    if isinstance(record, InvalidRecord):
        yield record.error
        return

    if not isinstance(record, Mapping):
        yield RecordError(
            field="record",
            code="invalid_type",
            message="Expected an object.",
        )
        return

    if profile == "text-rl":
        yield from _iter_text_rl_errors(record, require_fields)
        return

    yield from _iter_tokenized_sft_errors(record, max_length)


def validate_records(
    records: Iterable[tuple[Any, RecordLocation]],
    *,
    profile: Profile,
    require_fields: Iterable[str] = (),
    max_length: int | None = None,
    max_diagnostics: int = 20,
) -> ValidationReport:
    require_fields = tuple(require_fields)
    if profile not in VALID_PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    if max_diagnostics < 0:
        raise ValueError("max_diagnostics must be a non-negative integer")

    checked = 0
    valid = 0
    invalid = 0
    diagnostics: list[Diagnostic] = []
    diagnostics_truncated = False

    for record, location in records:
        checked += 1
        record_valid = True
        for error in iter_record_errors(
            record,
            profile,
            require_fields=require_fields,
            max_length=max_length,
        ):
            record_valid = False
            if len(diagnostics) == max_diagnostics:
                diagnostics_truncated = True
                break
            diagnostics.append(
                Diagnostic(
                    location=location,
                    field=error.field,
                    code=error.code,
                    message=error.message,
                )
            )
        if record_valid:
            valid += 1
        else:
            invalid += 1

    return ValidationReport(
        profile=profile,
        checked=checked,
        valid=valid,
        invalid=invalid,
        diagnostics=tuple(diagnostics),
        diagnostics_truncated=diagnostics_truncated,
    )


def _iter_text_rl_errors(
    record: Mapping[str, Any], require_fields: tuple[str, ...]
) -> Iterator[RecordError]:
    for field in require_fields:
        if field not in record:
            yield RecordError(
                field=field,
                code="missing_field",
                message="Required field is missing.",
            )

    if "messages" not in record:
        yield RecordError(
            field="messages",
            code="missing_field",
            message="Required field is missing.",
        )
        return

    messages = record["messages"]
    if not isinstance(messages, list):
        yield RecordError(
            field="messages",
            code="invalid_type",
            message="Expected a list.",
        )
        return
    if not messages:
        yield RecordError(
            field="messages",
            code="empty_list",
            message="Expected a non-empty list.",
        )
        return

    for index, message in enumerate(messages):
        message_field = f"messages[{index}]"
        if not isinstance(message, Mapping):
            yield RecordError(
                field=message_field,
                code="invalid_type",
                message="Expected an object.",
            )
            continue

        if "role" not in message:
            yield RecordError(
                field=f"{message_field}.role",
                code="missing_field",
                message="Required field is missing.",
            )
        elif not isinstance(message["role"], str):
            yield RecordError(
                field=f"{message_field}.role",
                code="invalid_type",
                message="Expected a string.",
            )
        elif message["role"] not in VALID_ROLES:
            yield RecordError(
                field=f"{message_field}.role",
                code="unsupported_role",
                message="Expected one of 'system', 'user', or 'assistant'.",
            )

        if "content" not in message:
            yield RecordError(
                field=f"{message_field}.content",
                code="missing_field",
                message="Required field is missing.",
            )
        elif not isinstance(message["content"], str):
            yield RecordError(
                field=f"{message_field}.content",
                code="invalid_type",
                message="Expected a string.",
            )


def _iter_tokenized_sft_errors(
    record: Mapping[str, Any], max_length: int | None
) -> Iterator[RecordError]:
    input_ids = record.get("input_ids")
    loss_mask = record.get("loss_mask")
    input_ids_valid = isinstance(input_ids, list)
    loss_mask_valid = isinstance(loss_mask, list)

    if "input_ids" not in record:
        yield RecordError(
            field="input_ids",
            code="missing_field",
            message="Required field is missing.",
        )
    elif not input_ids_valid:
        yield RecordError(
            field="input_ids",
            code="invalid_type",
            message="Expected a list.",
        )
    elif not input_ids:
        yield RecordError(
            field="input_ids",
            code="empty_list",
            message="Expected a non-empty list.",
        )

    if "loss_mask" not in record:
        yield RecordError(
            field="loss_mask",
            code="missing_field",
            message="Required field is missing.",
        )
    elif not loss_mask_valid:
        yield RecordError(
            field="loss_mask",
            code="invalid_type",
            message="Expected a list.",
        )
    elif not loss_mask:
        yield RecordError(
            field="loss_mask",
            code="empty_list",
            message="Expected a non-empty list.",
        )

    if not input_ids_valid or not loss_mask_valid:
        return

    if len(input_ids) != len(loss_mask):
        yield RecordError(
            field="loss_mask",
            code="length_mismatch",
            message=(
                f"Expected length {len(input_ids)} to match input_ids, "
                f"got {len(loss_mask)}."
            ),
        )

    if max_length is not None and len(input_ids) > max_length:
        yield RecordError(
            field="input_ids",
            code="max_length_exceeded",
            message=f"Expected at most {max_length} tokens.",
        )

    for index, token_id in enumerate(input_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            yield RecordError(
                field=f"input_ids[{index}]",
                code="invalid_token_id",
                message="Expected a non-negative integer.",
            )

    has_supervised_position = False
    for index, mask_value in enumerate(loss_mask):
        if not _is_valid_loss_mask_value(mask_value):
            yield RecordError(
                field=f"loss_mask[{index}]",
                code="invalid_loss_mask",
                message="Expected a boolean or numeric 0 or 1.",
            )
            continue
        if bool(mask_value):
            has_supervised_position = True

    if loss_mask and not has_supervised_position:
        yield RecordError(
            field="loss_mask",
            code="all_loss_mask_zero",
            message="Expected at least one supervised position.",
        )


def _is_valid_loss_mask_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int) and value in (0, 1):
        return True
    if isinstance(value, float) and value in (0.0, 1.0):
        return True
    return False
