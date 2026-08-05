# SPDX-License-Identifier: Apache-2.0
from areal.api.cli_args import (
    GenerationHyperparameters,
    GRPOConfig,
    NormConfig,
    PPOActorConfig,
)


def test_group_norm_group_size_derived_from_n_samples():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=8),
        actor=PPOActorConfig(
            reward_norm=NormConfig(mean_level="group", std_level="group"),
        ),
    )
    assert cfg.actor.reward_norm.group_size == 8


def test_mismatched_group_size_is_overridden():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            reward_norm=NormConfig(mean_level="group", group_size=7),
        ),
    )
    assert cfg.actor.reward_norm.group_size == 4


def test_batch_level_norm_group_size_untouched():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(
            adv_norm=NormConfig(mean_level="batch", std_level="batch", group_size=3),
        ),
    )
    assert cfg.actor.adv_norm.group_size == 3


def test_none_norm_is_safe():
    cfg = GRPOConfig(
        gconfig=GenerationHyperparameters(n_samples=4),
        actor=PPOActorConfig(reward_norm=None, adv_norm=None),
    )
    assert cfg.actor.reward_norm is None
    assert cfg.actor.adv_norm is None
