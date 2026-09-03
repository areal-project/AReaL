# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.reward import if_gap


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("<think>work</think>answer", "answer"),
        ("<think>unfinished", ""),
        ("plain answer", "plain answer"),
    ],
)
def test_extract_visible_answer(completion, expected):
    assert if_gap.extract_visible_answer(completion) == expected


@pytest.mark.parametrize(
    ("results", "expected"),
    [([True, False], 1.0 / 3.0), ([True, True], 1.0), ([], 0.0)],
)
def test_if_gap_reward_combines_pass_rate_and_strict_bonus(
    monkeypatch, results, expected
):
    monkeypatch.setattr(if_gap, "score_if_gap_spec", lambda *_: results)

    reward = if_gap.if_gap_reward_fn(
        prompt="",
        completions="answer",
        verify_engine="test",
        spec={},
    )

    assert reward == pytest.approx(expected)


def test_ifrl_loader_requires_explicit_root(monkeypatch):
    monkeypatch.delenv("IF_SYNTH_ROOT", raising=False)
    monkeypatch.setattr(if_gap, "_ifrl_engines", None)

    with pytest.raises(FileNotFoundError, match="not configured"):
        if_gap._load_ifrl_engines()
