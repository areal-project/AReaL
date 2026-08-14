"""Text sanitizers used by LifelongBench (LLB) runners."""

from __future__ import annotations


_LLB_ENV_PREAMBLE_MARKERS: tuple[str, ...] = (
    "I will ask you a question",  # db_bench environment prompt
    "help me operate a MySQL database",  # db_bench environment prompt (variant)
    "I will provide you with a task to perform on a Linux (Ubuntu) system.",  # os_interaction environment prompt
)


def _sanitize_lines_from_index(lines: list[str], start_idx: int) -> list[str]:
    """Strip the env preamble block starting at `start_idx` and return new lines."""
    if start_idx < 0 or start_idx >= len(lines):
        return lines
    if not lines[start_idx].lstrip().startswith("user:"):
        return lines
    if not any(m in lines[start_idx] for m in _LLB_ENV_PREAMBLE_MARKERS):
        return lines

    cut_from = start_idx
    cut_to = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].lstrip()
        if line.startswith("user:") and not any(
            marker in line for marker in _LLB_ENV_PREAMBLE_MARKERS
        ):
            cut_to = i
            break
        if line.startswith("assistant:"):
            cut_to = i
            break
    return lines[:cut_from] + lines[cut_to:]


def sanitize_llb_env_preamble(text: str) -> str:
    """Remove LLB DB/OS environment preamble when it appears as a repeated block."""
    if not text:
        return text

    lines = text.split("\n")
    if not lines:
        return text

    # First try: strip when the preamble is the leading block.
    first_nonempty = None
    for i, line in enumerate(lines):
        if line.strip():
            first_nonempty = i
            break
    if first_nonempty is None:
        return ""

    first = lines[first_nonempty].lstrip()
    if first.startswith("user:") and any(marker in first for marker in _LLB_ENV_PREAMBLE_MARKERS):
        stripped = _sanitize_lines_from_index(lines, first_nonempty)
        return "\n".join(stripped).strip()

    # Fallback: handle the case where trajectory is embedded inside a larger blob.
    for i, line in enumerate(lines):
        if line.lstrip().startswith("user:") and any(
            marker in line for marker in _LLB_ENV_PREAMBLE_MARKERS
        ):
            stripped = _sanitize_lines_from_index(lines, i)
            return "\n".join(stripped).strip()

    return text


__all__ = ["sanitize_llb_env_preamble"]

