# SPDX-License-Identifier: Apache-2.0

"""Geometry3K equivalence matrices, generated nesting, and malformed-answer cases."""

import random

import pytest

from areal.reward.geometry3k import extract_answer, geometry3k_reward_fn

# Each expected equivalence is independently known, rather than inferred from
# the verifier under test. Cover representations found in geometry solutions.
MATH_CASES = [
    ("integer zero", "0", "0"),
    ("negative", "-3", "-3"),
    ("signed decimal", "-0.125", "-\\frac{1}{8}"),
    ("thousands", "1,234", "1234"),
    ("fraction", "\\frac{2}{4}", "0.5"),
    ("display fraction", "\\dfrac{3}{4}", "0.75"),
    ("text fraction", "\\tfrac{3}{4}", "0.75"),
    ("nested fraction", "\\frac{\\frac{1}{2}}{\\frac{3}{4}}", "\\frac{2}{3}"),
    ("mixed fraction", "4\\frac{4}{11}", "\\frac{48}{11}"),
    ("negative mixed fraction", "-4\\frac{4}{11}", "-\\frac{48}{11}"),
    ("root simplify", "\\sqrt{12}", "2\\sqrt{3}"),
    ("nested root", "\\sqrt{\\sqrt{16}}", "2"),
    ("nth root prediction", "\\sqrt[3]{8}", "2"),
    ("nth root gold", "2", "\\sqrt[3]{8}"),
    ("fractional power", "8^{\\frac{1}{3}}", "2"),
    ("power", "2^3", "8"),
    ("negative power", "2^{-3}", "0.125"),
    ("pi coefficient", "2\\pi", "\\pi+\\pi"),
    ("pi fraction", "\\frac{\\pi}{2}", "0.5\\pi"),
    ("scientific", "1.2\\times10^3", "1200"),
    ("degree", "90^\\circ", "90"),
    ("degree braces", "90^{\\circ}", "90"),
    ("unit", "5\\text{ cm}", "5"),
    ("redundant braces", "{{5}}", "5"),
    ("parenthesized", "\\left(\\frac12\\right)", "0.5"),
    ("nested boxed whole", "\\boxed{1}+2", "3"),
    ("nested boxed numerator", "\\frac{\\boxed{1}}{2}", "0.5"),
    ("math-delimited gold", "5\\sqrt{3}", "$5\\sqrt{3}$"),
    ("display gold", "5\\sqrt{3}", "\\[5\\sqrt{3}\\]"),
    ("inline gold", "5\\sqrt{3}", "\\(5\\sqrt{3}\\)"),
]


CONTEXTS = [
    ("", ""),
    ("<think>Try [999] and 123.</think>", ". See [7], step 2."),
    (r"Earlier \boxed{999}, corrected to ", ""),
    ("Reasoning on line 1.\n", "\nTherefore this is final. 888"),
]


def _reward(completion, gold):
    return geometry3k_reward_fn("", completion, [], [], gold)


@pytest.mark.parametrize(
    "label,expression,gold", MATH_CASES, ids=[c[0] for c in MATH_CASES]
)
@pytest.mark.parametrize("prefix,suffix", CONTEXTS)
def test_geometry3k_equivalent_expression_and_wrong_answer(
    label, expression, gold, prefix, suffix
):
    """Equivalent boxed expressions earn credit; an unrelated scalar does not."""
    completion = prefix + r"\boxed{" + expression + "}" + suffix
    assert _reward(completion, gold) == 1.0, label
    wrong = prefix + r"\boxed{314159265}" + suffix
    assert _reward(wrong, gold) == 0.0, label


@pytest.mark.parametrize(
    "expression,wrong_gold",
    [
        (r"2\sqrt{3}", "3"),
        (r"\frac{1}{11}", "11"),
        (r"\frac{-7}{8}", "8"),
        (r"2^3", "3"),
        (r"2^{-3}", "-3"),
        (r"\sqrt[3]{8}", "3"),
        (r"1.2\times10^3", "3"),
        (r"4\frac{4}{11}", "11"),
        (r"\boxed{1}+2", "1"),
        (r"\frac{\boxed{1}}{2}", "1"),
    ],
)
def test_geometry3k_rejects_numeric_subexpression(expression, wrong_gold):
    """A denominator, radicand, exponent, or nested-box value is not the answer."""
    assert _reward(r"\boxed{" + expression + "}", wrong_gold) == 0.0


@pytest.mark.parametrize(
    "completion",
    [
        r"[2] \boxed{}",
        r"[2] \boxed{ }",
        r"[2] \boxed{",
        r"[2] \boxed{\frac{1}{2}",
        r"\boxed{2} corrected to \boxed{",
        r"\boxed{2} corrected to \boxed{}",
        r"\boxed{\boxed{2}",
        r"\boxed{\{2\}",
    ],
)
def test_geometry3k_malformed_final_box_does_not_earn_credit(completion):
    """Do not rescue a malformed final box using earlier answers or numbers."""
    assert extract_answer(completion) == ""
    assert _reward(completion, "2") == 0.0


@pytest.mark.parametrize("gold", ["", " ", None, r"\boxed{}", r"\boxed{"])
def test_geometry3k_missing_gold_never_earns_credit(gold):
    """An absent gold expression cannot authorize a correct reward."""
    assert _reward(r"\boxed{2}", gold) == 0.0


def _generated_expression(rng, depth):
    if depth == 0:
        return rng.choice(["0", "-7", "1.25", r"\pi", r"\{1,2\}", "[1,2]"])
    inner = _generated_expression(rng, depth - 1)
    return rng.choice(
        [
            r"\frac{" + inner + "}{3}",
            r"\sqrt{" + inner + "}",
            r"\sqrt[3]{" + inner + "}",
            r"\boxed{" + inner + "}+2",
            "{" + inner + "}^{2}",
        ]
    )


def test_geometry3k_generated_nested_expressions_and_truncations():
    """Exercise 1,000 deterministic expression trees in four noisy contexts."""
    rng = random.Random(1867)
    for index in range(1000):
        expression = _generated_expression(rng, rng.randrange(1, 9))
        for prefix, suffix in CONTEXTS:
            completion = prefix + "\\boxed {\n" + expression + "\n}" + suffix
            assert extract_answer(completion) == expression, (index, completion)
            # Omitting the outer closing brace must not expose an inner box.
            truncated = prefix + r"\boxed{" + expression
            assert extract_answer(truncated) == "", (index, truncated)


@pytest.mark.parametrize("depth", [32, 128, 1024])
def test_geometry3k_deep_balanced_braces_do_not_recurse(depth):
    """Deeply nested output is scanned without Python recursion or truncation."""
    expression = "{" * depth + "2" + "}" * depth
    assert extract_answer(r"\boxed{" + expression + "}") == expression
    assert extract_answer(r"\boxed{" + expression) == ""


@pytest.mark.parametrize("answer", ["0", "-3", "1,234.5", "0.5"])
def test_geometry3k_legacy_unboxed_formats_remain_supported(answer):
    """Legacy bracket and plain numerical answers retain their behavior."""
    expected = answer.replace(",", "")
    assert _reward("The answer is " + answer, expected) == 1.0
    assert _reward("The answer is [" + answer + "]", expected) == 1.0


@pytest.mark.parametrize(
    "completion,gold",
    [
        (r"\boxed{0.5}", r"\frac{\boxed{1}}{2}"),
        (r"\boxed{3}", r"1+\boxed{2}"),
        (r"\boxed{3}", r"\boxed{\boxed{1}+2}"),
        (r"\boxed{0.5}", r"[\frac{1}{2}]"),
        (r"\boxed{2}", r"[\sqrt[3]{8}]"),
        (r"\boxed{0.5}", r"$$\frac{1}{2}$$"),
    ],
)
def test_geometry3k_gold_wrappers_preserve_complete_expression(completion, gold):
    """Gold wrappers and inner formatting must not select a subexpression."""
    assert _reward(completion, gold) == 1.0
    assert _reward(r"\boxed{314159265}", gold) == 0.0


@pytest.mark.parametrize("gold", [r"\boxed{2", r"\boxed{\boxed{2}"])
def test_geometry3k_truncated_gold_is_not_rescued(gold):
    """Unclosed gold boxes must not silently become their last valid number."""
    assert _reward(r"\boxed{2}", gold) == 0.0


def test_geometry3k_escaped_box_marker_is_not_a_later_answer():
    """A literal backslash escape must not introduce a new answer box."""
    assert extract_answer(r"\boxed{2} literal \\boxed{3}") == "2"
