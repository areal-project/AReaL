"""Tests for the ``reuse_train_logp`` proximal-logp method.

``reuse_train_logp`` reuses the training forward-pass logprobs as the proximal
logp, skipping the extra decoupled-PPO forward. It requires
``ppo_n_minibatches == 1`` so the training forward still reflects the policy
that produced the rollout (with multiple minibatches the weights change between
steps, so the reused logprobs would no longer be the proximal policy).
"""

import pytest

from areal.api.cli_args import PPOActorConfig
from areal.utils.constants import (
    PROX_LOGP_METHOD_REUSE_TRAIN_LOGP,
    PROX_LOGP_METHODS_ALL,
    ProxLogpMethod,
)


def test_enum_includes_reuse_train_logp():
    assert ProxLogpMethod.REUSE_TRAIN_LOGP.value == "reuse_train_logp"
    assert ProxLogpMethod("reuse_train_logp") is ProxLogpMethod.REUSE_TRAIN_LOGP
    assert "reuse_train_logp" in PROX_LOGP_METHODS_ALL
    assert PROX_LOGP_METHOD_REUSE_TRAIN_LOGP == "reuse_train_logp"


def test_reuse_train_logp_skips_forward_pass():
    # It must skip the extra proximal forward, like loglinear does.
    assert ProxLogpMethod.REUSE_TRAIN_LOGP.skips_forward_pass()


def test_reuse_train_logp_allows_single_minibatch():
    config = PPOActorConfig(
        backend="fsdp:d1",
        prox_logp_method="reuse_train_logp",
        ppo_n_minibatches=1,
    )
    assert config.prox_logp_method == "reuse_train_logp"
    assert config.ppo_n_minibatches == 1


def test_reuse_train_logp_rejects_multiple_minibatches():
    with pytest.raises(ValueError, match="ppo_n_minibatches=1"):
        PPOActorConfig(
            backend="fsdp:d1",
            prox_logp_method="reuse_train_logp",
            ppo_n_minibatches=2,
        )


def test_other_methods_unaffected_by_minibatch_count():
    # The constraint only applies to reuse_train_logp.
    config = PPOActorConfig(
        backend="fsdp:d1",
        prox_logp_method="recompute",
        ppo_n_minibatches=4,
    )
    assert config.ppo_n_minibatches == 4
