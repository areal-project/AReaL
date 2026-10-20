from types import SimpleNamespace
from unittest.mock import Mock

import flask
import pytest

from areal.engine.awex.adapters.megatron_adapter import AwexMegatronAdapter
from areal.v2.training_service.worker.awex import create_awex_blueprint


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
