import warnings

import pytest

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    PPOConfig,
    SGLangConfig,
)


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


def test_ppo_config_warns_when_eval_sampling_seed_set_without_deterministic_inference():
    """eval_gconfig can carry its own sampling_seed independent of gconfig (e.g. a
    fixed seed for held-out eval while training rollouts are unseeded); the check
    must not miss it just because gconfig itself has no seed set."""
    with pytest.warns(UserWarning, match="sampling_seed is set but"):
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            eval_gconfig=GenerationHyperparameters(sampling_seed=42),
        )


def test_ppo_config_does_not_warn_when_eval_flags_are_consistent():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            eval_gconfig=GenerationHyperparameters(sampling_seed=42),
            sglang=SGLangConfig(enable_deterministic_inference=True),
        )

    assert not any("sampling_seed is set but" in str(w.message) for w in caught)


def test_ppo_config_warns_without_crashing_when_sglang_is_none():
    """sglang is not Optional and the YAML/CLI loader rejects `sglang: null`, but
    direct Python construction (PPOConfig(sglang=None, ...)) bypasses that loader and
    is not type-checked at runtime, so this must not raise AttributeError."""
    with pytest.warns(UserWarning, match="sampling_seed is set but"):
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            gconfig=GenerationHyperparameters(sampling_seed=42),
            sglang=None,
        )


def test_ppo_config_raises_when_sampling_seed_set_with_vllm_backend():
    """vLLM has no sampling_seed support; both VLLMBackend and VLLMBridgeBackend
    raise NotImplementedError on the first generation request, so fail fast at
    config-construction time instead of after server launch and model load."""
    with pytest.raises(ValueError, match="does not support sampling_seed"):
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            gconfig=GenerationHyperparameters(sampling_seed=42),
            rollout=InferenceEngineConfig(backend="vllm:d2t4"),
        )


def test_ppo_config_does_not_raise_when_vllm_backend_without_sampling_seed():
    PPOConfig(
        experiment_name="exp",
        trial_name="trial",
        rollout=InferenceEngineConfig(backend="vllm:d2t4"),
    )


def test_ppo_config_does_not_raise_when_sglang_backend_with_sampling_seed():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            gconfig=GenerationHyperparameters(sampling_seed=42),
            rollout=InferenceEngineConfig(backend="sglang:d2t4"),
            sglang=SGLangConfig(enable_deterministic_inference=True),
        )


def test_ppo_config_does_not_raise_when_rollout_backend_unset():
    """Default InferenceEngineConfig().backend is the OmegaConf MISSING sentinel
    ('???'), not a real backend string; must not crash or misfire as vLLM."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            gconfig=GenerationHyperparameters(sampling_seed=42),
        )
