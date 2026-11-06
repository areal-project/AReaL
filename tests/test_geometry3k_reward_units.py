# SPDX-License-Identifier: Apache-2.0

"""Geometry3K unit rejection, unit-free golds, and partial-number controls."""

import pytest

from areal.reward.geometry3k import geometry3k_reward_fn


def _reward(completion, gold):
    return geometry3k_reward_fn("", completion, [], [], gold)


@pytest.mark.parametrize(
    "completion,gold",
    [
        (r"\boxed{5 centimeters}", "5"),
        (r"\boxed{5 in}", "5"),
        ("[90 cm^2]", "90"),
    ],
)
def test_geometry3k_reviewed_unit_suffix_regressions(completion, gold):
    """The unit-free answer policy rejects the formerly accepted unit suffixes."""
    assert _reward(completion, gold) == 0.0


@pytest.mark.parametrize("wrapper", [r"\boxed{%s}", "[%s]"])
@pytest.mark.parametrize(
    "expression,gold,wrong_gold",
    [
        ("5", "5", "6"),
        (r"2\sqrt{3}", r"\sqrt{12}", "3"),
        (r"\frac{1}{11}", r"\frac{2}{22}", "11"),
        ("2^3", "8", "3"),
        (r"2\pi", r"\pi+\pi", "2"),
    ],
)
@pytest.mark.parametrize(
    "unit",
    [
        "centimeters",
        "centimetres",
        "millimeter",
        "kilometres",
        "m",
        "in",
        "inches",
        "ft",
        "feet",
        "yards",
        "degrees",
        "radians",
        "units",
        "cm^2",
        "cm^{2}",
        "cm ^ { 3 }",
        "cm²",
        "cm³",
        "square centimeters",
        "sq. in",
        "cubic feet",
        "cm squared",
        "cm cubed",
        "CM",
        r"\,cm^2",
        r"\quad centimeters",
    ],
)
def test_geometry3k_units_are_rejected(wrapper, expression, gold, wrong_gold, unit):
    """A unit-bearing prediction or gold must not be reduced to a bare quantity."""
    completion = wrapper % (expression + " " + unit)
    assert _reward(completion, gold) == 0.0
    assert _reward(completion, wrong_gold) == 0.0
    assert _reward(wrapper % expression, gold + " " + unit) == 0.0


@pytest.mark.parametrize(
    "expression,wrong_gold",
    [
        ("5 x", "5"),
        ("5 xyz", "5"),
        ("5 centimetersx", "5"),
        ("5 inchworm", "5"),
        ("5 cm^4", "5"),
        ("5 in + 2", "7"),
        ("5 + in", "5"),
        ("5 * in", "5"),
        ("5 / in", "5"),
        (r"5 \times in", "5"),
        (r"2\sqrt{3} centimeters", "3"),
        (r"\frac{1}{11} in", "11"),
        (r"\sqrt[3]{8} cm^2", "3"),
    ],
)
def test_geometry3k_unit_handling_does_not_enable_numeric_fallback(
    expression, wrong_gold
):
    """Unknown suffixes and expressions cannot earn credit for one number."""
    assert _reward(r"\boxed{" + expression + "}", wrong_gold) == 0.0


@pytest.mark.parametrize(
    "completion",
    [r"\boxed{5 in} corrected to \boxed{}", r"\boxed{5 centimeters"],
)
def test_geometry3k_units_do_not_rescue_malformed_final_boxes(completion):
    """Unit rejection must not bypass final-box validation."""
    assert _reward(completion, "5") == 0.0


@pytest.mark.parametrize("wrapper", [r"\boxed{%s}", "[%s]"])
@pytest.mark.parametrize(
    "expression",
    [
        "5cm",
        "5 m",
        "5 apples",
        "5°",
        "5º",
        "5'",
        '5"',
        "5′",
        "5″",
        r"5^\circ",
        r"5^{\circ}",
        r"5\degree",
        r"5\text{cm}",
        r"5\mathrm{cm}",
        r"5\operatorname{cm}",
        r"5\text{ cm}^{2}",
        r"5\text{ cm}^2",
        r"5\,\mathrm{cm}^{-1}",
        r"\SI{5}{cm}",
        r"5\cm",
        r"\frac{5\text{cm}}{1\text{cm}}",
        "5 cm / cm",
        "5 + 0 cm",
        r"5 + 0\text{cm}",
        r"{5\text{cm}}",
        "5 centimetres",
        "5㎝",
        "5㎟",
        "5㎠",
        "5{''}",
        '5{"}',
    ],
)
def test_geometry3k_rejects_units_before_verifier_normalization(wrapper, expression):
    """Units must not disappear through formatting, cancellation, or parsing."""
    assert _reward(wrapper % expression, "5") == 0.0


@pytest.mark.parametrize(
    "completion,gold",
    [
        (r"\boxed{5 in}", "5 cm"),
        (r"\boxed{5 cm}", "5 m"),
        (r"\boxed{5 cm^2}", "5 cm^3"),
        (r"\boxed{5 cm}", "5 cm"),
        (r"\boxed{5}", r"\boxed{5\text{cm}}"),
        ("The answer is 5 cm^2", "2"),
        ("The answer is 5 cm", "5"),
        ("The answer is 5 apples", "5"),
        (r"The answer is 5\cm", "5"),
    ],
)
def test_geometry3k_does_not_compare_or_discard_units(completion, gold):
    """Neither matching nor mismatching measurements are unit-free answers."""
    assert _reward(completion, gold) == 0.0


@pytest.mark.parametrize(
    "gold",
    [
        "a + b = 90",
        r"\angle 8 \cong \angle 2",
        "not enough information",
        r"\angle 1 \cong \angle 4",
        "m + n - 180",
        "20 y^2 - 10 y",
        "x < 60",
        r"\angle 2 and \angle 6",
        "angles 2,3, 6, and 7",
        "A D and B E",
        r"\angle 3 \cong \angle Z",
    ],
)
def test_geometry3k_retains_actual_unit_free_symbolic_and_text_golds(gold):
    """All eleven non-numeric dataset golds remain valid, including variable m."""
    assert _reward(r"\boxed{" + gold + "}", gold) == 1.0
    assert _reward(r"\boxed{314159265}", gold) == 0.0


def test_geometry3k_checks_selected_answer_without_penalizing_units_in_reasoning():
    """Only the selected answer must be unit-free, not the surrounding reasoning."""
    assert _reward(r"The length is 5 cm. Final: \boxed{5}", "5") == 1.0
    assert _reward(r"Earlier \boxed{5 cm}, corrected to \boxed{5}", "5") == 1.0


def test_geometry3k_worker_does_not_change_shared_verifier_unit_policy():
    """Geometry3K's strict extraction must not mutate the shared math worker."""
    from areal.reward import get_math_verify_worker

    shared = get_math_verify_worker()
    assert shared.verify(r"$5\text{cm}$", "$5$") == 1.0
    assert _reward(r"\boxed{5\text{cm}}", "5") == 0.0
    assert shared.verify(r"$5\text{cm}$", "$5$") == 1.0
