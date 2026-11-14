# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from examples.swe.utils import SWEPPOConfig

from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils.dynamic_import import bind_filter_kwargs
from areal.v2.inference_service.controller.controller import RolloutControllerV2

FILTER = "examples.swe.filter_function.filter_by_accuracy"
OPTIONS = {"min_accuracy": 0.4, "max_accuracy": 0.8}


def _sample(correct):
    return {"original_rewards": torch.tensor([1.0] * correct + [0.0] * (5 - correct))}


def test_accuracy_filter_options_load_from_yaml_and_cli_override():
    config = OmegaConf.merge(
        OmegaConf.structured(SWEPPOConfig),
        OmegaConf.create(
            f"should_accept_fn: {FILTER}\nshould_accept_kwargs:\n  min_accuracy: 0.2\n  max_accuracy: 1.0\n"
        ),
        OmegaConf.from_dotlist(
            [
                "should_accept_kwargs.min_accuracy=0.4",
                "should_accept_kwargs.max_accuracy=0.8",
            ]
        ),
    )
    # Supply unrelated required experiment/timer metadata for a complete config.
    for key in OmegaConf.missing_keys(config):
        OmegaConf.update(config, key, "test")
    loaded = OmegaConf.to_object(config)
    assert loaded.should_accept_kwargs == OPTIONS
    engine = object.__new__(RemoteInfEngine)
    fn = engine._resolve_should_accept_fn(
        loaded.should_accept_fn, loaded.should_accept_kwargs
    )
    assert not fn(_sample(2))  # lower boundary: accepted by the old default
    assert fn(_sample(3))
    assert not fn(_sample(4))  # upper boundary: accepted by the old default


@pytest.mark.parametrize(
    "options",
    [
        {"min_accuracy": -0.1},
        {"min_accuracy": 0.9, "max_accuracy": 0.4},
        {"max_accuracy": 1.1},
        {"max_accuracy": float("nan")},
        {"correct_reward_threshold": float("inf")},
    ],
)
def test_accuracy_filter_invalid_options_fail_during_config_creation(options):
    with pytest.raises(ValueError):
        SWEPPOConfig(should_accept_fn=FILTER, should_accept_kwargs=options)


def test_filter_kwargs_require_a_function_and_valid_keyword_names():
    with pytest.raises(ValueError, match="requires should_accept_fn"):
        SWEPPOConfig(should_accept_kwargs=OPTIONS)
    with pytest.raises(ValueError, match="requires should_accept_fn"):
        bind_filter_kwargs(None, OPTIONS)
    engine = object.__new__(RemoteInfEngine)
    with pytest.raises(TypeError):
        engine._resolve_should_accept_fn(FILTER, {"min_accurracy": 0.4})
    with pytest.raises(TypeError):
        engine._resolve_should_accept_fn(FILTER, {"sample": {}})
    assert engine._resolve_should_accept_fn(None, {}) is None


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("method", ["submit", "prepare_batch"])
def test_worker_binds_serialized_filter_options_before_execution(version, method):
    cls = RemoteInfEngine if version == "v1" else RolloutControllerV2
    engine = object.__new__(cls)
    engine._resolve_workflow = Mock(return_value=object())
    executor = Mock()
    executor.prepare_batch.return_value = []
    if version == "v2":
        engine._ensure_initialized = Mock()
        engine._gateway_addr = "test"
        engine._workflow_executor = executor
    else:
        engine.workflow_executor = executor
    options = deserialize_value(serialize_value(OPTIONS))
    getattr(engine, method)(
        [], workflow="unused", should_accept_fn=FILTER, should_accept_kwargs=options
    )
    fn = getattr(engine.workflow_executor, method).call_args.kwargs["should_accept_fn"]
    assert not fn(_sample(2))
    assert fn(_sample(3))
    assert not fn(_sample(4))


@pytest.mark.parametrize("method", ["submit", "prepare_batch"])
def test_controller_preserves_filter_options_through_worker_rpc(method):
    from tests.test_rollout_controller import (
        MockInferenceEngine,
        MockScheduler,
        create_test_config,
    )

    scheduler = MockScheduler()
    controller = RolloutController(
        inf_engine=MockInferenceEngine,
        config=create_test_config(consumer_batch_size=1),
        scheduler=scheduler,
    )
    controller.initialize(role="rollout", server_args={})
    try:
        if method == "submit":
            controller.submit(
                {},
                workflow="tests.utils.TestWorkflow",
                should_accept_fn=FILTER,
                should_accept_kwargs=OPTIONS,
            )
            controller.wait(count=1, timeout=5)
        else:
            from torchdata.stateful_dataloader import StatefulDataLoader

            loader = StatefulDataLoader([{}, {}], batch_size=1, collate_fn=list)
            controller.prepare_batch(
                loader,
                workflow="tests.utils.TestWorkflow",
                should_accept_fn=FILTER,
                should_accept_kwargs=OPTIONS,
            )
        submissions = [
            kwargs for _, name, _, kwargs in scheduler.engine_calls if name == "submit"
        ]
        assert submissions
        worker = object.__new__(RemoteInfEngine)
        for kwargs in submissions:
            assert kwargs["should_accept_kwargs"] == OPTIONS
            fn = worker._resolve_should_accept_fn(
                kwargs["should_accept_fn"], kwargs["should_accept_kwargs"]
            )
            assert not fn(_sample(2))
            assert fn(_sample(3))
    finally:
        controller.destroy()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_trainer_passes_dynamic_filter_options_to_batch_preparation(
    monkeypatch, version
):
    from tests.test_trainer_eval_before_train import (
        _build_ppo_trainer,
        _disable_timing_contexts,
        _StopAfterFirstUpdate,
    )

    from areal.trainer import rl_trainer

    _disable_timing_contexts(monkeypatch, rl_trainer)
    trainer = _build_ppo_trainer([])
    trainer.config.rollout._version = version
    trainer.actor.prepare_batch = Mock(side_effect=_StopAfterFirstUpdate)
    with pytest.raises(_StopAfterFirstUpdate):
        trainer.train(
            workflow=object(), dynamic_filter_fn=FILTER, dynamic_filter_kwargs=OPTIONS
        )
    kwargs = trainer.actor.prepare_batch.call_args.kwargs
    assert kwargs["should_accept_fn"] == FILTER
    assert kwargs["should_accept_kwargs"] == OPTIONS
