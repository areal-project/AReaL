import pytest


@pytest.mark.parametrize("reward", [0.0, 0.25, 1.0])
def test_baseline_identity_reward_import_preserves_scores(reward):
    from areal.utils.dynamic_import import import_from_string

    transform = import_from_string("examples.swe.reward_transforms.identity_reward")
    assert transform(reward, {}, reward_threshold=0.98) == reward


@pytest.mark.parametrize("reward", [-0.1, 1.1, float("nan"), float("inf")])
def test_baseline_identity_reward_rejects_invalid_scores(reward):
    from examples.swe.reward_transforms import identity_reward

    with pytest.raises(ValueError, match="finite"):
        identity_reward(reward, {})
