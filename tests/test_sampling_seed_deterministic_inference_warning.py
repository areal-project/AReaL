import warnings

import pytest

from areal.api.cli_args import GenerationHyperparameters, PPOConfig, SGLangConfig


def test_ppo_config_warns_when_sampling_seed_set_without_deterministic_inference():
    """SGLang silently ignores per-request sampling_seed unless the server runs with
    --enable-deterministic-inference (SGLangConfig.enable_deterministic_inference).
    Since both fields live on the same PPOConfig, catch the common misconfiguration
    at config-construction time rather than leaving it a silent no-op discoverable
    only by reading SGLang internals."""
    with pytest.warns(UserWarning, match="sampling_seed is set but"):
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            gconfig=GenerationHyperparameters(sampling_seed=42),
        )


def test_ppo_config_does_not_warn_when_flags_are_consistent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            gconfig=GenerationHyperparameters(sampling_seed=42),
            sglang=SGLangConfig(enable_deterministic_inference=True),
        )

    assert not any("sampling_seed is set but" in str(w.message) for w in caught)


def test_ppo_config_does_not_warn_when_sampling_seed_unset():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PPOConfig(experiment_name="exp", trial_name="trial")

    assert not any("sampling_seed is set but" in str(w.message) for w in caught)
