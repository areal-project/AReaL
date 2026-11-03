# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.reward.geometry3k import extract_answer, geometry3k_reward_fn


@pytest.mark.parametrize(
    "completion, expected",
    [
        (r"\boxed{90}", "90"),
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{5\sqrt{3}}", r"5\sqrt{3}"),
        (r"The answer is \boxed{5\sqrt{3}}.", r"5\sqrt{3}"),
        (r"The answer is \boxed{2\sqrt{3}}.", r"2\sqrt{3}"),
        (r"The answer is \boxed{\frac{1}{11}}.", r"\frac{1}{11}"),
        (r"\boxed{\frac{1+\sqrt{5}}{2}}", r"\frac{1+\sqrt{5}}{2}"),
        (r"\boxed{\sqrt[3]{8}}", r"\sqrt[3]{8}"),
        (r"\boxed{4\frac{4}{11}}", r"4\frac{4}{11}"),
        (r"\boxed{2^3}", "2^3"),
        (r"\boxed { 90 }", "90"),
        ("\\boxed{\n90\n}", "90"),
        (r"\boxed{\{1, 2\}}", r"\{1, 2\}"),
        (r"<think>Try [60].</think>\boxed{90}", "90"),
        (r"\boxed{90}. See [1], step 2.", "90"),
        (r"\boxed{60} corrected to \boxed{90}", "90"),
        (r"\boxed{90} corrected to \boxed{}", ""),
        (r"\boxed{90} corrected to \boxed{\frac{1}{2}", ""),
        (r"\boxed{   }", ""),
        (r"\boxed{\frac{1}{2}", ""),
        ("The angle is [90].", "90"),
        ("The angle is 90 degrees.", "90"),
        ("The answer is -1,234.5", "-1234.5"),
        ("No answer", ""),
        ("", ""),
    ],
)
def test_extract_answer_preserves_final_expression(completion, expected):
    """Prefer complete boxed expressions while keeping legacy fallbacks."""
    assert extract_answer(completion) == expected


@pytest.mark.parametrize(
    "answer, expected",
    [
        (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"[5\sqrt{3}]", r"5\sqrt{3}"),
        (r"5\sqrt{3}", ""),
    ],
)
def test_extract_answer_without_number_fallback(answer, expected):
    """Gold extraction must preserve expressions rather than take a last number."""
    assert extract_answer(answer, use_last_number=False) == expected


@pytest.mark.parametrize(
    "completion, answer, expected",
    [
        # Exact reproductions from issue #1867.
        (r"The answer is \boxed{5\sqrt{3}}.", r"5 \sqrt { 3 }", 1.0),
        (r"The answer is \boxed{2\sqrt{3}}.", "3", 0.0),
        (r"The answer is \boxed{\frac{1}{11}}.", "11", 0.0),
        (r"\boxed{\frac{1}{2}}", "0.5", 1.0),
        (r"\boxed{\frac{1}{2}}", "2", 0.0),
        (r"\boxed{5\sqrt{3}}", r"5 \sqrt { 3 }", 1.0),
        (r"\boxed{5\sqrt{3}}", "3", 0.0),
        (r"\boxed{4\frac{4}{11}}", r"4 \frac { 4 } { 11 }", 1.0),
        (r"\boxed{2^3}", "8", 1.0),
        (r"\boxed{2^3}", "3", 0.0),
        (r"<think>Try [60].</think>\boxed{90}. See step 2.", "90", 1.0),
        (r"\boxed{60} corrected to \boxed{90}", "60", 0.0),
        (r"\boxed{90} corrected to \boxed{60}", "60", 1.0),
        (r"\boxed{}", "90", 0.0),
        (r"\boxed{\frac{1}{2}", "2", 0.0),
        (r"\boxed{\frac{1}{2}}", r"\boxed{0.5}", 1.0),
    ],
)
def test_geometry3k_reward_grades_whole_boxed_answer(completion, answer, expected):
    """Exercise real math verification for correct and misleading last numbers."""
    reward = geometry3k_reward_fn(
        prompt=r"Put the final answer in \boxed{}.",
        completions=completion,
        prompt_ids=[],
        completion_ids=[],
        answer=answer,
    )
    assert reward == expected
