"""LLB prompt utilities.

This module centralizes LifelongBench (LLB) prompt construction so the system
prompt stays consistent across runners/entrypoints.
"""

from __future__ import annotations


DEFAULT_SYSTEM_PROMPT = """
You are an execution-focused AI agent solving database and operating-system tasks.

You may receive a [Retrieved Memory Context] block with past experiences from similar problems.
These are **references for learning**, not guaranteed solutions:
- [MEMORY TYPE] SUCCESS_PROCEDURE: A successful approach from a similar task—learn the pattern.
- [MEMORY TYPE] FAILURE_REFLECTION: A failed attempt with lessons—avoid similar mistakes.

Use the memories as inspiration, but always analyze your current task independently and
adapt your approach based on its specific requirements.
""".strip()


LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT = """
STRICT OUTPUT FORMAT (LLB-DB, do not violate):
1) After your reasoning, include exactly ONE action line:
   - Action: Operation
   - Action: Answer
2) If Action: Operation, put exactly ONE SQL statement in the FIRST fenced code block using ```sql, on a single line. Do not add any extra text after that block.
3) If Action: Answer, include `Final Answer: ...` on the next line and do not add extra text after that.
4) ANSWER FORMAT for Final Answer (SELECT queries):
   - Return comma-separated tuples in round brackets: (val1, val2), (val3, val4), ...
   - Never use square brackets or list notation around the tuples.
   - Preserve Decimal values exactly as returned by the database.
   - Row order must match your SQL query's ORDER BY.
""".strip()


LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT = """
STRICT OUTPUT FORMAT (LLB-OS, do not violate):
1) After your reasoning, include exactly ONE action line:
   - Act: bash
   - Act: finish
2) If Act: bash, the next lines MUST be a ```bash fenced code block with your Bash commands. Do not include any other code blocks.
3) If Act: finish, it must be the last line (no code blocks, no extra text).
4) Do NOT use `Action:` in OS tasks (use `Act:` only).
""".strip()


# Backward-compatible alias (historically this constant existed and was DB-oriented).
LLB_STRICT_OUTPUT_FORMAT_CONSTRAINT = LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT


def llb_strict_output_constraint_for_task(task: str) -> str | None:
    """Return the task-aligned strict output format constraint block."""
    t = (task or "").strip().lower()
    if t in ("db", "db_bench", "db_bench_tts", "db_bench_resume"):
        return LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT
    if t in ("os", "os_interaction", "os_interaction_tts", "os_interaction_resume"):
        return LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT
    return None


def build_llb_system_prompt(*, task: str, base_prompt: str | None = None) -> str:
    """Build the LLB system prompt for a given task (db/os), aligning constraints."""
    constraint = llb_strict_output_constraint_for_task(task)
    base = (base_prompt if base_prompt is not None else DEFAULT_SYSTEM_PROMPT).strip()
    if not constraint:
        return base

    # If a strict-format block already exists but is for a different task, strip it.
    if "STRICT OUTPUT FORMAT" in base:
        if "STRICT OUTPUT FORMAT (LLB-DB" in base or "STRICT OUTPUT FORMAT (LLB-OS" in base:
            # Keep only when it matches this task.
            if constraint.splitlines()[0] in base:
                return base
            idx = base.rfind("STRICT OUTPUT FORMAT")
            if idx != -1:
                base = base[:idx].rstrip()
        else:
            # Legacy marker: we only ever append this block at the end, so drop the tail.
            idx = base.rfind("STRICT OUTPUT FORMAT")
            if idx != -1 and (len(base) - idx) < 1200:
                base = base[:idx].rstrip()

    if not base:
        return constraint
    return (base + "\n\n" + constraint).strip()


def strip_llb_strict_output_format_block(text: str) -> str:
    """Strip a trailing LLB strict-output-format block from a prompt, if present."""
    s = (text or "").strip()
    if not s:
        return s

    idx = s.rfind("STRICT OUTPUT FORMAT")
    if idx == -1:
        return s

    # Conservative: only strip when the marker is near the end.
    if (len(s) - idx) > 2000:
        return s

    return s[:idx].rstrip()


def build_llb_prompt_with_memory(
    *,
    task: str,
    base_prompt: str | None = None,
    memory_context: str | None = None,
) -> str:
    """Build the final LLB prompt with memory injected.

    Ordering:
      1) system prompt
      2) [Retrieved Memory Context] (optional)
      3) STRICT OUTPUT FORMAT block at the very end (task-aligned)
    """
    base = (base_prompt if base_prompt is not None else DEFAULT_SYSTEM_PROMPT).strip()
    base = strip_llb_strict_output_format_block(base)

    mem = (memory_context or "").strip()
    combined = base
    if mem:
        combined = f"{base}\n\n{mem}".strip()

    return build_llb_system_prompt(task=task, base_prompt=combined)


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LLB_DB_STRICT_OUTPUT_FORMAT_CONSTRAINT",
    "LLB_OS_STRICT_OUTPUT_FORMAT_CONSTRAINT",
    "LLB_STRICT_OUTPUT_FORMAT_CONSTRAINT",
    "llb_strict_output_constraint_for_task",
    "build_llb_system_prompt",
    "strip_llb_strict_output_format_block",
    "build_llb_prompt_with_memory",
]

