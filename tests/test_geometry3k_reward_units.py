# SPDX-License-Identifier: Apache-2.0

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
    """Plain units must not turn a correct quantity into symbolic variables."""
    assert _reward(completion, gold) == 1.0


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
def test_geometry3k_units_preserve_complete_quantity(
    wrapper, expression, gold, wrong_gold, unit
):
    """Strip known suffixes on either side without accepting a subexpression."""
    completion = wrapper % (expression + " " + unit)
    assert _reward(completion, gold) == 1.0
    assert _reward(completion, wrong_gold) == 0.0
    assert _reward(wrapper % expression, gold + " " + unit) == 1.0


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
    """Unit normalization must not bypass final-box validation."""
    assert _reward(completion, "5") == 0.0
