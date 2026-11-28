import pytest

from areal.reward import clevr_count_70k_reward_fn
from areal.utils.dynamic_import import import_from_string


def test_clevr_non_string_answer_scored_correctly():
    """A non-string answer (e.g. an int) must be coerced and scored, not crash.
    The sibling reward fns str()-coerce both inputs; clevr did not, so a matching
    completion scored as a dropped trajectory instead of 1.0."""
    reward = clevr_count_70k_reward_fn(
        prompt="",
        completions="[3]",
        prompt_ids=[],
        completion_ids=[],
        answer=3,
    )
    assert reward == 1.0


@pytest.mark.parametrize(
    ("completion", "answer", "expected"),
    [
        ("[ 3 ]", "3", 1.0),
        ("First estimate [3]. Final answer: [ 4 ].", "3", 0.0),
        ("First estimate [4]. Final answer: [3].", "3", 1.0),
        ("First estimate [3]. Final answer: [four].", "3", 0.0),
        ("First estimate [3]. Final answer: [].", "3", 0.0),
        ("First estimate [3]. Final answer: [4", "3", 0.0),
        ("The answer is 3.", "3", 0.0),
        ("[four]", "four", 0.0),
        ("[2.5]", "2.5", 1.0),
    ],
)
def test_clevr_scores_final_bracketed_answer(completion, answer, expected):
    """Only the final bracketed count determines the reward."""
    assert clevr_count_70k_reward_fn("", completion, [], [], answer) == expected


def test_clevr_grpo_reward_path():
    """The GRPO workflow can load the shared scorer by its configured path."""
    reward_fn = import_from_string(
        "examples.vlm.clevr_count_70k_grpo.clevr_count_70k_reward_fn"
    )
    assert reward_fn is clevr_count_70k_reward_fn
