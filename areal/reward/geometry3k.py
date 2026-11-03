# SPDX-License-Identifier: Apache-2.0

import re

from areal.utils import logging

from . import get_math_verify_worker

logger = logging.getLogger("Geometry3KReward")

# Only explicit, separated geometry-unit suffixes are annotations. Do not
# remove arbitrary words or symbols, which may be part of the answer.
_GEOMETRY_UNIT_SUFFIX = re.compile(
    # An annotation must follow a quantity, not an operator such as `5 + in`.
    r"(?:(?<=[\d})\]])|(?<=\\pi))"
    r"(?:\s|\\[,; ]|\\(?:quad|qquad)\b)+"
    r"(?:(?:square|cubic|sq\.?|cu\.?)\s+)?"
    r"(?:mm|cm|dm|km|m|(?:milli|centi|deci|kilo)?met(?:er|re)s?"
    r"|inches|inch|in|feet|foot|ft|yards?|yd|miles?|mi"
    r"|degrees?|deg|radians?|rad|units?)"
    r"(?:\s*\^\s*(?:[23]|\{\s*[23]\s*\})|[²³]|\s+(?:squared|cubed))?"
    r"\s*$",
    re.IGNORECASE,
)


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
    # Validate boxes without selecting one: gold is already the full answer.
    # Returning boxed content would turn 5 + \boxed{2} into 2. Leave valid
    # boxes intact for the grouping-preserving normalization before verify.
    if _extract_boxed_answer(text) == "":
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
    prompt, completions: str, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """Grade a decoded completion string against a complete gold expression."""
    try:
        # Container reprs escape backslashes and can hide boxes from extraction.
        if not isinstance(completions, str):
            return 0.0
        sol = extract_answer(completions)
        ans = _extract_gold_answer(answer)

        if not sol or not ans:
            return 0.0

        # Once the final expression is selected, nested boxes are just
        # grouping. Otherwise math-verify can prioritize an inner box over
        # the surrounding fraction or sum.
        sol = re.sub(r"\\boxed\s*(?=\{)", "", sol)
        ans = re.sub(r"\\boxed\s*(?=\{)", "", ans)

        # Bare unit letters would otherwise become LaTeX variables. Keep the
        # entire quantity: a raw-text fallback could accept its last number.
        sol = _GEOMETRY_UNIT_SUFFIX.sub("", sol).strip()
        ans = _GEOMETRY_UNIT_SUFFIX.sub("", ans).strip()

        worker = get_math_verify_worker()
        # Wrap in $...$ so math-verify parses bare LaTeX as one expression
        # instead of extracting its last number.
        return worker.verify(f"${sol}$", f"${ans}$")
    except Exception:
        logger.warning("Exception in geometry3k_reward_fn", exc_info=True)
        return 0.0
