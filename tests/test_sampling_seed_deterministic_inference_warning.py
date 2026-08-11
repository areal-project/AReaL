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


def test_ppo_config_raises_on_grouped_deterministic_rollouts_without_seed():
    """With deterministic inference on but no per-request seed, SGLang gives every
    seedless request the same default seed, so a group's n_samples>1 same-prompt
    rollouts collapse to identical completions and zero out GRPO's advantage. This
    combination has no working use and the flag is new and default-off, so fail fast
    at config time rather than burning a whole training run on a degenerate config."""
    with pytest.raises(ValueError, match="collapse to identical completions"):
        PPOConfig(
            experiment_name="exp",
            trial_name="trial",
            gconfig=GenerationHyperparameters(n_samples=8),
            sglang=SGLangConfig(enable_deterministic_inference=True),
        )


def test_ppo_config_does_not_raise_on_grouped_deterministic_rollouts_with_seed():
    """A seed IS set, the user's signal they intend distinct per-rollout seeds, so
    the collapse guard must not fire."""
    PPOConfig(
        experiment_name="exp",
        trial_name="trial",
        gconfig=GenerationHyperparameters(n_samples=8, sampling_seed=42),
        sglang=SGLangConfig(enable_deterministic_inference=True),
    )


def test_ppo_config_does_not_raise_on_deterministic_single_sample_without_seed():
    """n_samples == 1 is not a group, so there is no intra-group diversity to
    collapse; the guard must not fire (it would otherwise block every non-grouped
    run that enables deterministic inference)."""
    PPOConfig(
        experiment_name="exp",
        trial_name="trial",
        gconfig=GenerationHyperparameters(n_samples=1),
        sglang=SGLangConfig(enable_deterministic_inference=True),
    )


def test_ppo_config_does_not_raise_on_grouped_deterministic_greedy_without_seed():
    """Under greedy decoding the request builders decode at temperature 0.0, so the
    group collapses regardless of any seed and a per-request seed would not change
    it; the seed-focused guard is scoped to stochastic sampling and must not fire."""
    PPOConfig(
        experiment_name="exp",
        trial_name="trial",
        gconfig=GenerationHyperparameters(n_samples=8, greedy=True),
        sglang=SGLangConfig(enable_deterministic_inference=True),
    )


def test_ppo_config_does_not_raise_on_grouped_deterministic_temperature_zero():
    """temperature == 0 is likewise seed-independent argmax decoding; the guard is
    scoped to temperature > 0 and must not fire."""
    PPOConfig(
        experiment_name="exp",
        trial_name="trial",
        gconfig=GenerationHyperparameters(n_samples=8, temperature=0.0),
        sglang=SGLangConfig(enable_deterministic_inference=True),
    )


def test_ppo_config_does_not_raise_on_grouped_deterministic_vllm_backend():
    """enable_deterministic_inference is an SGLang-only flag and inert on vLLM, so
    the SGLang-worded collapse guard must not fire for a vLLM backend even when the
    flag is (pointlessly) set; a vLLM run does not hit SGLang's default-seed path."""
    PPOConfig(
        experiment_name="exp",
        trial_name="trial",
        gconfig=GenerationHyperparameters(n_samples=8),
        rollout=InferenceEngineConfig(backend="vllm:d2t4"),
        sglang=SGLangConfig(enable_deterministic_inference=True),
    )
