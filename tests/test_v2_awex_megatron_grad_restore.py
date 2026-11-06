# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import flask
import pytest

from areal.engine.awex.adapters.megatron_adapter import AwexMegatronAdapter
from areal.engine.megatron_engine import MegatronEngine
from areal.v2.training_service.worker.awex import create_awex_blueprint


class _StopAfterZeroGrad(Exception):
    pass


def test_train_batch_restores_residency_grad_buffers_before_zero_grad(monkeypatch):
    events = []
    monkeypatch.setattr(
        "areal.engine.megatron_engine.record_training_batch",
        lambda *args: nullcontext(),
    )

    def zero_grad():
        events.append("zero_grad")
        raise _StopAfterZeroGrad

    engine = SimpleNamespace(
        _weight_residency=SimpleNamespace(
            ensure_grad_buffers=lambda: events.append("restore_grad_buffers")
        ),
        _ensure_ready=lambda: events.append("ensure_ready"),
        optimizer_zero_grad=zero_grad,
        _normalize_batch_input=lambda batch: (batch, None),
        hf_config=SimpleNamespace(),
    )
    with pytest.raises(_StopAfterZeroGrad):
        MegatronEngine.train_batch(
            engine,
            input_={},
            loss_fn=lambda *_args: None,
            loss_weight_fn=lambda *_args: None,
        )
    assert events == ["ensure_ready", "restore_grad_buffers", "zero_grad"]


def test_colocate_enable_shares_engine_residency_and_teardown_preserves_it():
    state = SimpleNamespace(released_tags=frozenset({"weights", "optimizer"}))
    engine = SimpleNamespace(
        _awex_adapter=None, _ensure_weight_residency=Mock(return_value=state)
    )
    adapter = AwexMegatronAdapter(engine)
    assert engine._awex_adapter is None

    adapter.enable_colocate_memory_management()
    adapter.teardown_colocate_weight_update()

    assert engine._awex_adapter is adapter
    assert adapter.residency is state
    assert adapter._released_tags == {"weights", "optimizer"}


def test_failed_publish_restores_initial_offloaded_weight_state():
    state = SimpleNamespace(is_released=Mock(side_effect=[True, False]))
    adapter = AwexMegatronAdapter(SimpleNamespace())
    adapter._residency = state
    adapter._colocate_init_fingerprint = ("pair",)
    adapter._execute_meta_colocate_weight_update = Mock(
        side_effect=RuntimeError("publish")
    )
    adapter.release_memory = Mock()

    with pytest.raises(RuntimeError, match="publish"):
        adapter.execute_colocate_weight_update(1)

    adapter.release_memory.assert_called_once_with(tags=["weights"])


def test_release_before_meta_init_uses_one_shared_residency(monkeypatch):
    released = set()
    residency = SimpleNamespace(
        released_tags=released,
        release_memory=lambda tags: released.update(tags),
        resume_memory=lambda tags: released.difference_update(tags),
    )
    engine = SimpleNamespace(cpu_group="cpu", _awex_adapter=None)
    engine._ensure_weight_residency = lambda: residency
    engine.hf_config = SimpleNamespace(architectures=["Qwen3ForCausalLM"])
    adapter = AwexMegatronAdapter(engine)
    monkeypatch.setattr(
        "areal.v2.training_service.worker.awex._create_training_adapter",
        lambda engine: adapter,
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr("awex.meta.meta_server.MetaServerClient", Mock())

    def run_endpoint(_name, action, return_result=True):
        result = action()
        return flask.jsonify({"result": result} if return_result else {"status": "ok"})

    app = flask.Flask(__name__)
    app.register_blueprint(
        create_awex_blueprint(
            flask_module=flask,
            get_engine=lambda: engine,
            submit_to_engine_thread=lambda name, action: action(),
            run_endpoint=run_endpoint,
        )
    )
    client = app.test_client()

    response = client.post(
        "/awex/release_memory", json={"tags": ["weights", "optimizer"]}
    )
    assert response.status_code == 200
    assert adapter.residency is residency
    assert released == {"weights", "optimizer"}
    response = client.post(
        "/awex/init_colocate_weight_update",
        json=dict(
            pair_name="pair",
            meta_server_addr="192.0.2.1:1234",
            transfer_rank=2,
            infer_world_size=2,
            train_world_size=2,
            num_engines=1,
            timeout_s=30,
        ),
    )
    assert response.status_code == 200
    assert adapter.residency is residency
    assert released == {"weights", "optimizer"}
    response = client.post(
        "/awex/resume_memory", json={"tags": ["weights", "optimizer"]}
    )
    assert response.status_code == 200
    assert not released


@pytest.mark.parametrize("role", ["ref", "critic", "teacher"])
def test_auxiliary_bootstrap_initializes_residency_before_offload(role):
    from areal.trainer.rl_trainer import PPOTrainer

    events = []
    engine = Mock()
    engine.initialize.side_effect = lambda **kwargs: events.append("initialize")
    engine.init_weight_residency_adapter.side_effect = lambda: events.append(
        "residency"
    )
    config = SimpleNamespace(_version="v2", backend="megatron:d1")
    trainer = object.__new__(PPOTrainer)
    trainer.ref = trainer.critic = trainer.teacher = None
    trainer.config = SimpleNamespace(ref=None, critic=None, teacher=None)
    trainer._should_offload_ref = trainer._should_offload_critic = False
    trainer._should_offload_teacher = False
    setattr(trainer, role, engine)
    setattr(trainer, f"_should_offload_{role}", True)
    if role == "teacher":
        trainer.config.teacher = SimpleNamespace(engine_type="train", train=config)
    else:
        setattr(trainer.config, role, config)
    trainer._offload_model = lambda actual, **kwargs: events.append("offload")

    trainer._initialize_colocated_auxiliary_engines({"addr": None, "ft_spec": None})

    assert events == ["initialize", "residency", "offload"]
    engine.initialize.assert_called_once_with(addr=None, ft_spec=None, role=role)
