# SPDX-License-Identifier: Apache-2.0

import re

from areal.utils import logging

from . import get_math_verify_worker

logger = logging.getLogger("Geometry3KReward")


def _extract_boxed_answer(text: str) -> str | None:
    """Return the last outer box, empty for an unfinished box, or None if absent."""
    answer = None
    start = 0
    depth = 0
    # Scan forward so an inner box cannot override its enclosing expression.
    # Escaped characters (including literal braces) are not delimiters.
    for token in re.finditer(r"\\boxed\s*\{|\\.|[{}]", text):
        value = token.group()
        if value.startswith(r"\boxed"):
            if depth == 0:
                start = token.end()
            depth += 1
        elif depth:
            if value == "{":
                depth += 1
            elif value == "}":
                depth -= 1
                if depth == 0:
                    answer = text[start : token.start()].strip()
    # Do not rescue an unfinished final box using earlier answers or numbers.
    return "" if depth else answer


def extract_answer(pred_str, use_last_number=True):
    # The dataset prompt asks for \boxed{}. Preserve the whole expression.
    boxed = _extract_boxed_answer(pred_str)
    if boxed is not None:
        return boxed

    matches = re.findall(r"\[([^\]]+)\]", pred_str)
    if matches:
        return matches[-1]

    # Fallback to last number if no bracket format found
    if use_last_number:
        pattern = r"-?\d*\.?\d+"
        nums = re.findall(pattern, pred_str.replace(",", ""))
        if nums:
            return nums[-1]

    return ""


def _extract_gold_answer(answer) -> str:
    """Keep gold expressions whole, removing only an outer answer wrapper."""
    if answer is None:
        return ""
    text = str(answer).strip()
    boxed = _extract_boxed_answer(text)
    if boxed == "":
        return ""

    # Do not mistake the index in \sqrt[3]{8} for a bracketed gold answer.
    for left, right in [
        (r"\[", r"\]"),
        (r"\(", r"\)"),
        ("$$", "$$"),
        ("$", "$"),
        ("[", "]"),
    ]:
        if (
            text.startswith(left)
            and text.endswith(right)
            and len(text) >= len(left) + len(right)
        ):
            return text[len(left) : -len(right)].strip()
    return text


def geometry3k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        sol = extract_answer(str(completions))
        ans = _extract_gold_answer(answer)

        if not sol or not ans:
            return 0.0

        # Once the final expression is selected, nested boxes are just
        # grouping. Otherwise math-verify can prioritize an inner box over
        # the surrounding fraction or sum.
        sol = re.sub(r"\\boxed\s*(?=\{)", "", sol)
        ans = re.sub(r"\\boxed\s*(?=\{)", "", ans)

        worker = get_math_verify_worker()
        # Wrap in $...$ so math-verify parses bare LaTeX as one expression
        # instead of extracting its last number.
        return worker.verify(f"${sol}$", f"${ans}$")
    except Exception:
        logger.warning("Exception in geometry3k_reward_fn", exc_info=True)
        return 0.0
