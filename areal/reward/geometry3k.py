# SPDX-License-Identifier: Apache-2.0

import re
import unicodedata
from dataclasses import replace
from functools import lru_cache

from areal.utils import logging

from . import MathVerifyWorker

logger = logging.getLogger("Geometry3KReward")

# Single-letter m is ambiguous: it is a variable when present in the gold.
_UNIT_WORD = re.compile(
    r"(?:mm|cm|dm|km|(?:milli|centi|deci|kilo)?met(?:er|re)s?"
    r"|inches|inch|in|feet|foot|ft|yards?|yd|miles?|mi"
    r"|degrees?|deg|radians?|rad|units?|square|cubic|squared|cubed)",
    re.IGNORECASE,
)
_UNIT_SYMBOL = re.compile(r"[°º′″℃℉'\"]|\\(?:circ|degree|degrees|prime|si|SI)\b")
_MATH_WORDS = {
    "e",
    "i",
    "pi",
    "π",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "log",
    "ln",
    "exp",
    "abs",
}
_MATH_COMMANDS = set(
    "frac dfrac tfrac cfrac sqrt pi left right text mathrm mathit mathbf "
    "boxed angle cong times cdot div pm mp le ge leq geq ne neq lt gt approx "
    "quad qquad operatorname ln log exp sin cos tan infty abs overline "
    "underline hat widehat sum prod int lim vec".split()
)


def _answer_words(text: str) -> set[str]:
    """Read identifiers and prose, including text inside LaTeX font wrappers."""
    # Compatibility characters such as ㎝ must not conceal a unit annotation.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(
        r"\\([a-zA-Z]+)",
        lambda match: " " if match[1] in _MATH_COMMANDS else " " + match[1] + " ",
        text,
    )
    return set(re.findall(r"[^\W\d_]+", text))


def _is_unit_free(text: str, gold_words: set[str]) -> bool:
    # Check before parsing: units can cancel algebraically or be silently
    # removed by the parser, including degree marks and text-wrapped units.
    if _UNIT_SYMBOL.search(text) or _UNIT_SYMBOL.search(
        unicodedata.normalize("NFKC", text)
    ):
        return False
    words = _answer_words(text)
    return not any(_UNIT_WORD.fullmatch(word) for word in words) and words <= (
        gold_words | _MATH_WORDS
    )


@lru_cache(maxsize=1)
def _get_geometry3k_worker() -> MathVerifyWorker:
    """Keep unit removal and unanchored numeric extraction out of Geometry3K."""
    from math_verify.parser import LatexExtractionConfig

    worker = MathVerifyWorker()
    latex = LatexExtractionConfig()
    targets = (
        replace(
            latex,
            normalization_config=replace(
                latex.normalization_config, units=False, boxed="none"
            ),
        ),
    )
    worker.pred_extraction_target = targets
    worker.gold_extraction_target = targets
    return worker


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
    """Grade a unit-free completion against a unit-free mathematical/text gold."""
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

        gold_words = _answer_words(ans)
        if not _is_unit_free(sol, gold_words) or not _is_unit_free(ans, gold_words):
            return 0.0

        # The legacy number fallback loses suffixes. Inspect the numeric tail
        # before accepting it, so unboxed "5 cm^2" cannot become the number 2.
        if _extract_boxed_answer(completions) is None and not re.search(
            r"\[([^\]]+)\]", completions
        ):
            number = re.search(r"-?\d", completions)
            if number and not _is_unit_free(completions[number.start() :], gold_words):
                return 0.0

        worker = _get_geometry3k_worker()
        # Wrap in $...$ so math-verify parses bare LaTeX as one expression
        # instead of extracting its last number.
        return worker.verify(f"${sol}$", f"${ans}$")
    except Exception:
        logger.warning("Exception in geometry3k_reward_fn", exc_info=True)
        return 0.0
