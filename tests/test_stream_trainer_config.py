# SPDX-License-Identifier: Apache-2.0

"""Validate the deliberately narrow first streamed-training configuration."""

import pytest

from areal.api.cli_args import (
    FP8EngineConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    NormConfig,
    PPOActorConfig,
    PPOConfig,
    PRMScorerConfig,
    SchedulingSpec,
    SchedulingStrategy,
    StreamTrainerConfig,
    TrainDatasetConfig,
)


def _stream_config():
    return PPOConfig(
        experiment_name="stream-test",
        trial_name="trial",
        enable_offload=True,
        stream_trainer=StreamTrainerConfig(enabled=True),
        gconfig=GenerationHyperparameters(n_samples=4),
        train_dataset=TrainDatasetConfig(batch_size=2),
        actor=PPOActorConfig(
            backend="megatron:d2t1p1",
            weight_update_mode="awex",
            disable_dropout=True,
            ppo_n_minibatches=1,
            kl_ctl=0.0,
            scheduling_spec=(SchedulingSpec(port_count=3),),
            reward_norm=NormConfig(mean_level="group", std_level="group", group_size=4),
            adv_norm=NormConfig(mean_level="group", std_level=None, group_size=4),
        ),
        rollout=InferenceEngineConfig(
            backend="sglang:d2t1p1",
            consumer_batch_size=2,
            max_head_offpolicyness=0,
            scheduling_strategy=SchedulingStrategy(
                type="colocation", target="actor", fork=True
            ),
        ),
    )


def test_stream_trainer_config_default_preserves_disabled_path():
    """Existing PPO configurations do not opt in to stream-only restrictions."""
    assert not StreamTrainerConfig().enabled
    assert not PPOConfig().stream_trainer.enabled


def test_stream_trainer_config_matching_dp2_awex_accepts_group_statistics():
    """Complete-group reward and advantage normalization needs no future batch."""
    config = _stream_config()
    assert config.stream_trainer.enabled
    assert config.actor.reward_norm.mean_level == "group"
    assert config.actor.adv_norm.mean_level == "group"
    assert config.rollout.consumer_batch_size == config.train_dataset.batch_size == 2


@pytest.mark.parametrize("field", ["mean_level", "std_level"])
def test_stream_trainer_config_batch_advantage_statistics_rejected(field):
    """Early gradients cannot use statistics from groups still being generated."""
    config = _stream_config()
    setattr(config.actor.adv_norm, field, "batch")

    with pytest.raises(
        ValueError, match=r"actor\.adv_norm cannot use batch statistics"
    ):
        config.__post_init__()


def test_stream_trainer_config_fp8_training_rejected():
    """FP8 stateful scaling is outside the fixed-policy streaming contract."""
    config = _stream_config()
    config.actor.megatron.fp8_config = FP8EngineConfig()
    with pytest.raises(ValueError, match="FP8 training is unsupported"):
        config.__post_init__()


def test_stream_trainer_config_enabled_prm_scorer_rejected():
    config = _stream_config()
    config.rollout.agent.prm.scorers.append(PRMScorerConfig())
    with pytest.raises(ValueError, match="PRM advantage shaping is unsupported"):
        config.__post_init__()


def test_stream_trainer_config_d2t4_actor_d4t2_rollout_accepted():
    """Two inference replicas may jointly release one actor TP replica."""
    config = _stream_config()
    config.actor.backend = "megatron:d2t4p1"
    config.rollout.backend = "sglang:d4t2p1"

    config.__post_init__()

    assert config.stream_trainer.enabled


@pytest.mark.parametrize(
    "actor_backend,rollout_backend,message",
    [
        ("megatron:d2t1p1", "sglang:d2t2p1", "same total GPU count"),
        ("megatron:d4t2p1", "sglang:d2t4p1", "actor TP must be a multiple"),
    ],
)
def test_stream_trainer_config_incompatible_topology_rejected(
    actor_backend, rollout_backend, message
):
    config = _stream_config()
    config.actor.backend = actor_backend
    config.rollout.backend = rollout_backend
    config.train_dataset.batch_size = config.rollout.consumer_batch_size = 4
    with pytest.raises(ValueError, match=message):
        config.__post_init__()


def test_stream_trainer_config_consumer_batch_mismatch_rejected():
    """One strict-policy batch must agree with the accepted-capacity boundary."""
    config = _stream_config()
    config.rollout.consumer_batch_size = 3
    with pytest.raises(
        ValueError, match=r"consumer_batch_size must match train_dataset\.batch_size"
    ):
        config.__post_init__()


def test_stream_trainer_config_batch_smaller_than_dp_rejected():
    """The initial implementation requires at least one group per actor replica."""
    config = _stream_config()
    config.rollout.consumer_batch_size = config.train_dataset.batch_size = 1
    with pytest.raises(ValueError, match="batch_size must be at least actor DP"):
        config.__post_init__()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), True, "0.1"])
def test_stream_trainer_config_invalid_poll_interval_rejected(interval):
    """Invalid polling must fail at configuration time instead of a busy loop."""
    with pytest.raises(ValueError, match="poll_interval must be finite and positive"):
        StreamTrainerConfig(enabled=True, poll_interval=interval)
