"""
Unified LLM output sanitization.

Single place to clean raw LLM responses before:
- Storing as memory content (trajectory/script)
- Injecting into prompts as memory context
- Extracting code from model responses

Handles DeepSeek-R1 <think> reasoning blocks and markdown fences.
"""

import re


def sanitize_llm_output(text: str) -> str:
    """
    Sanitize raw LLM output for safe downstream use.

    1) Remove all <think>...</think> blocks (including nested/multiline).
    2) If entire remaining output is wrapped in a single markdown code fence,
       remove the outer fence and preserve inner content.
    3) Trim leading/trailing whitespace.
    4) Return empty string for None or empty input.
    """
    if text is None:
        return ""

    s = str(text).strip()
    if not s:
        return ""

    # Remove <think>...</think> blocks (handles nested via stack)
    open_tag = "<think>"
    close_tag = "</think>"
    result_parts = []
    i = 0
    depth = 0
    n = len(s)

    while i < n:
        if s.startswith(open_tag, i):
            depth += 1
            i += len(open_tag)
            continue
        if s.startswith(close_tag, i):
            if depth > 0:
                depth -= 1
            i += len(close_tag)
            continue
        if depth == 0:
            result_parts.append(s[i])
        i += 1

    s = "".join(result_parts).strip()
    if not s:
        return ""

    # Remove a single outer markdown code fence if it wraps the entire content
    fence_pattern = re.compile(r"^\s*```[^\n]*\n([\s\S]*?)\n?```\s*$")
    m = fence_pattern.match(s)
    if m:
        s = m.group(1).strip()

    return s
