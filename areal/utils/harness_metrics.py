# SPDX-License-Identifier: Apache-2.0
"""Shared Harness classification for episode and retained-row diagnostics."""

from collections.abc import Mapping
from typing import Any

HARNESS_OUTCOME_METRIC_CODES = frozenset(
    {
        "AGENT_MAX_TURNS_EXCEEDED",
        "AGENT_RUN_TIMEOUT",
        "AUTONOMOUS_INCOMPLETE_NO_SHIP",
        "GAMEAGENT_RUN_FAILED",
        "INVALID_TOOL_JSON",
        "LLM_CONTEXT_WINDOW_EXCEEDED",
        "LLM_OUTPUT_TOKEN_LIMIT_EXCEEDED",
        "LLM_RESPONSE_FAILED",
        "LLM_RESPONSE_TIMEOUT",
    }
)
HARNESS_OUTCOME_CODES = ("OTHER", *sorted(HARNESS_OUTCOME_METRIC_CODES))
HARNESS_OUTCOME_IDS = {
    code: index + 1 for index, code in enumerate(HARNESS_OUTCOME_CODES)
}


def classify_harness_outcome(
    metadata: Mapping[str, Any] | None, *, require_status: bool = False
) -> int:
    """Return -1 for unknown, 0 for no Harness failure, or a failure code ID.

    A non-Harness failure is not necessarily task success. Legacy workflows may
    omit status; consumers requiring explicit status classify that as unknown.
    Unknown failure codes map to OTHER without creating unbounded metric names.
    """
    status = (metadata or {}).get("arena_status")
    if require_status and (not isinstance(status, str) or not status.strip()):
        return -1
    if status == "HARNESS_FAILED":
        code = metadata.get("harness_outcome_code")
        return HARNESS_OUTCOME_IDS.get(code, 1) if isinstance(code, str) else 1
    return 0
