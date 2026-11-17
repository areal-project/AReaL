# SPDX-License-Identifier: Apache-2.0

"""CPU coverage of v1 vLLM startup orchestration, without launching engines."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.api.cli_args import InferenceEngineConfig, PPOActorConfig, PPOConfig
from areal.trainer import rl_trainer
from areal.trainer.rl_trainer import PPOTrainer


@pytest.fixture
def runtime(monkeypatch):
    actor = Mock(data_parallel_rank=0, data_parallel_world_size=1)
    actor.get_version.return_value = 0
    rollout = Mock(server_infos=[object()])
    state = SimpleNamespace(
        config=PPOConfig(
            experiment_name="initial-weights-test",
            trial_name="unit",
            actor=PPOActorConfig(backend="fsdp:d1", kl_ctl=0.0),
            rollout=InferenceEngineConfig(backend="vllm:d1", agent=None),
            valid_dataset=None,
        ),
        actor=actor,
        rollout=rollout,
        recovery=Mock(),
    )
    state.recovery.load.return_value = None
    monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
    monkeypatch.setattr(rl_trainer, "prepare_awex_runtime", Mock(return_value=None))
    monkeypatch.setattr(
        rl_trainer, "load_hf_processor_and_tokenizer", Mock(return_value=(None, Mock()))
    )
    monkeypatch.setattr(rl_trainer.logging, "setup_file_logging", Mock())
    monkeypatch.setattr(rl_trainer.seeding, "set_random_seed", Mock())
    for name in ("Evaluator", "Saver", "StatsLogger"):
        monkeypatch.setattr(rl_trainer, name, Mock())
    monkeypatch.setattr(rl_trainer, "RecoverHandler", Mock(return_value=state.recovery))
    monkeypatch.setattr(PPOTrainer, "_create_train_engine", Mock(return_value=actor))
    monkeypatch.setattr(PPOTrainer, "_create_dataloader", Mock(return_value=[{}]))
    for name in (
        "_init_scheduler",
        "_amend_xccl_weight_update_envvar",
        "_config_perf_tracer",
        "_apply_initial_offload_policy",
    ):
        monkeypatch.setattr(PPOTrainer, name, Mock())
    monkeypatch.setattr(
        rl_trainer.RemotevLLMEngine,
        "as_controller",
        Mock(side_effect=[rollout, Mock()]),
    )
    return state


@pytest.mark.parametrize(
    "source,recovered", [("checkpoint", False), ("actor", False), ("actor", True)]
)
def test_initial_weight_source_controls_loading_and_sync(runtime, source, recovered):
    """Check default loading, actor initialization, and recovery without resync."""
    assert runtime.config.rollout.initial_weight_source == "checkpoint"
    runtime.config.rollout.initial_weight_source = source
    if recovered:
        runtime.recovery.load.return_value = SimpleNamespace(
            last_step_info=SimpleNamespace(global_step=4)
        )

    PPOTrainer(runtime.config, train_dataset=[{}])

    expected_format = "dummy" if source == "actor" else "auto"
    assert (
        runtime.rollout.initialize.call_args.kwargs["server_args"]["load_format"]
        == expected_format
    )
    assert runtime.config.vllm.load_format == "auto"
    if source == "actor" and not recovered:
        runtime.actor.update_weights.assert_called_once()
        assert runtime.actor.update_weights.call_args.args[0].version == 0
        runtime.rollout.set_version.assert_called_once_with(0)
    else:
        runtime.actor.update_weights.assert_not_called()
        runtime.rollout.set_version.assert_not_called()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("weight_update_mode", "disk", "not 'disk'"),
        ("use_lora", True, "adapter-only updates cannot initialize"),
    ],
)
def test_invalid_actor_source_fails_before_model_initialization(
    runtime, field, value, message
):
    """Reject disk updates and adapter-only initialization before loading models."""
    runtime.config.rollout.initial_weight_source = "actor"
    setattr(runtime.config.actor, field, value)

    with pytest.raises(ValueError, match=message):
        PPOTrainer(runtime.config, train_dataset=[{}])

    PPOTrainer._create_train_engine.assert_not_called()
    runtime.actor.initialize.assert_not_called()
    runtime.rollout.initialize.assert_not_called()
