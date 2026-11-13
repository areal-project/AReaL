# SPDX-License-Identifier: Apache-2.0
"""Application HTTP bytes, kept separate from CSR and NIC wire accounting."""

import json

import torch

from areal.utils import stats_tracker

HTTP_COUNTER_KEYS = frozenset(
    f"sampling_mask/http/{key}"
    for key in (
        "generate_response_body_bytes",
        "generate_responses",
        "reference_json_value_bytes",
    )
)


def _record(**values: int) -> None:
    stats_tracker.get("sampling_mask/http").stat_compact(
        torch.ones(1, dtype=torch.bool, device="cpu"),
        stats_tracker.ReduceType.SUM,
        **{
            key: torch.tensor([value], dtype=torch.float64, device="cpu")
            for key, value in values.items()
        },
    )


def record_response_body(size: int) -> None:
    """Count successfully received, decoded HTTP bodies, including non-mask data."""
    _record(generate_response_body_bytes=size, generate_responses=1)


def record_reference_json(references: list[list[int]]) -> None:
    """Integer-array JSON value bytes, matching SGLang's compact ORJSON encoding.

    Excludes the field name, punctuation surrounding the value, and HTTP headers.
    It includes repeated references sent by aborted/resubmitted requests.
    """
    _record(
        reference_json_value_bytes=len(
            json.dumps(references, separators=(",", ":")).encode("ascii")
        )
    )
