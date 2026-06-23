from areal.reward import clevr_count_70k_reward_fn


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
