from types import SimpleNamespace

import pytest
import torch
from flask import Flask

from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.rpc import rtensor
from areal.infra.rpc.guard import engine_blueprint
from areal.infra.rpc.guard.app import GuardState
from areal.infra.rpc.guard.engine_blueprint import engine_bp
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.workflow_executor import WorkflowContractFailure, _RolloutResult


@pytest.fixture
def client():
    app = Flask(__name__)
    # The blueprint calls get_state(), which looks for this config key
    state = GuardState()
    app.config["guard_state"] = state
    app.register_blueprint(engine_bp)

    with app.test_client() as client:
        yield client


def test_create_engine_empty_string(client):
    """Ensure empty strings are rejected (Functional parity with old manual check)."""
    resp = client.post("/create_engine", json={"engine": "", "engine_name": "test"})
    assert resp.status_code == 400
    # Pydantic errors are returned in the 'error' key per your route logic
    assert "error" in resp.get_json()


def test_create_engine_missing_fields(client):
    """Ensure missing required fields are caught by Pydantic."""
    resp = client.post(
        "/create_engine", json={"engine_name": "test"}
    )  # missing 'engine'
    assert resp.status_code == 400


def test_call_engine_missing_method(client):
    """Ensure missing method is rejected."""
    resp = client.post("/call", json={"engine_name": "actor/0"})
    assert resp.status_code == 400


def test_set_env_invalid_json(client):
    """Ensure malformed JSON or invalid types are rejected."""
    # Sending a string where an object is expected for 'env'
    resp = client.post("/set_env", json={"env": "not-a-dict"})
    assert resp.status_code == 400


@pytest.mark.parametrize(
    (
        "method",
        "cpu_staged_rpc_methods",
        "expected_localize",
        "expected_remotize",
    ),
    [
        ("echo", frozenset(), False, False),
        ("echo", frozenset({"echo"}), True, False),
        ("wait_for_task", frozenset(), False, True),
        ("_wait_for_task_result", frozenset(), False, True),
    ],
)
def test_call_engine_scopes_alias_preservation_to_supported_v1_boundaries(
    client,
    monkeypatch,
    method,
    cpu_staged_rpc_methods,
    expected_localize,
    expected_remotize,
):
    """Only CPU-staged inputs and grouped rollout outputs preserve aliases."""

    class FakeEngine:
        def __init__(self):
            self.cpu_staged_rpc_methods = cpu_staged_rpc_methods

        def echo(self, value):
            return value

        def wait_for_task(self, value):
            return value

        def _wait_for_task_result(self, value):
            return value

    localize_calls = []
    remotize_calls = []

    def fake_localize(obj, *, preserve_tensor_aliases=False):
        localize_calls.append(preserve_tensor_aliases)
        return obj

    def fake_remotize(obj, node_addr, *, preserve_tensor_aliases=False):
        remotize_calls.append(preserve_tensor_aliases)
        return obj

    monkeypatch.setitem(engine_blueprint._engines, "test", FakeEngine())
    monkeypatch.setattr(
        engine_blueprint, "_submit_to_engine_thread", lambda _name, func: func()
    )
    monkeypatch.setattr(
        engine_blueprint.RTensor, "localize", staticmethod(fake_localize)
    )
    monkeypatch.setattr(
        engine_blueprint.RTensor, "remotize", staticmethod(fake_remotize)
    )

    response = client.post(
        "/call",
        json={
            "method": method,
            "engine_name": "test",
            "args": ["payload"],
            "kwargs": {},
        },
    )

    assert response.status_code == 200
    assert localize_calls == [expected_localize]
    assert remotize_calls == [expected_remotize]


@pytest.mark.parametrize("outcome", ["success", "rejected", "contract_failure"])
def test_raw_rollout_result_preserves_remote_tensor_transport(
    client, monkeypatch, outcome
):
    image = torch.ones(1, 4)
    trajectory = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]]),
        "multi_modal_input": [{"pixel_values": image}, {"pixel_values": image}],
    }
    result = {
        "success": _RolloutResult(task_id=7, trajectory=trajectory),
        "rejected": None,
        "contract_failure": WorkflowContractFailure("invalid group"),
    }[outcome]

    class Engine:
        _wait_for_task_result = RemoteInfEngine._wait_for_task_result
        workflow_executor = SimpleNamespace(_wait_for_task_result=lambda *args: result)

    stored = []

    def store_tensor(tensor):
        stored.append(tensor)
        return str(len(stored))

    monkeypatch.setattr(rtensor, "_backend", SimpleNamespace(store=store_tensor))
    monkeypatch.setitem(engine_blueprint._engines, "test", Engine())
    monkeypatch.setattr(
        engine_blueprint, "_submit_to_engine_thread", lambda _, fn: fn()
    )
    response = client.post(
        "/call",
        json={"method": "_wait_for_task_result", "engine_name": "test", "args": [7]},
    )
    assert response.status_code == 200, response.get_json()
    payload = deserialize_value(response.get_json()["result"])
    if outcome != "success":
        assert payload == result
        assert stored == []
        return
    assert isinstance(payload["input_ids"], RTensor)
    assert payload["input_ids"].shape == (2, 2)
    first, second = payload["multi_modal_input"]
    assert first["pixel_values"].shard == second["pixel_values"].shard
    assert len(stored) == 3


@pytest.mark.parametrize(
    ("is_vision_model", "cpu_staged", "expected"),
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_call_engine_only_opts_vision_cpu_staging_into_alias_broadcast(
    client, monkeypatch, is_vision_model, cpu_staged, expected
):
    """Text engines keep the existing broadcast even when CPU staging is enabled."""
    from types import SimpleNamespace

    engine = SimpleNamespace(
        is_vision_model=is_vision_model,
        cpu_staged_rpc_methods={"echo"} if cpu_staged else set(),
        current_data_parallel_head=lambda: 0,
        echo=lambda value: value,
    )
    calls = []

    def broadcast(value, **kwargs):
        calls.append(kwargs["preserve_tensor_aliases"])
        return value

    monkeypatch.setitem(engine_blueprint._engines, "test", engine)
    monkeypatch.setattr(
        engine_blueprint, "_submit_to_engine_thread", lambda name, fn: fn()
    )
    monkeypatch.setattr(
        engine_blueprint, "_should_broadcast_payload", lambda **kw: True
    )
    monkeypatch.setattr(
        engine_blueprint, "resolve_broadcast_target", lambda *a: (None, "cpu")
    )
    monkeypatch.setattr(engine_blueprint, "broadcast_tensor_container", broadcast)
    monkeypatch.setattr(engine_blueprint.RTensor, "localize", lambda obj, **kw: obj)
    monkeypatch.setattr(engine_blueprint.RTensor, "remotize", lambda obj, *a, **kw: obj)

    response = client.post(
        "/call", json={"method": "echo", "engine_name": "test", "args": ["text"]}
    )

    assert response.status_code == 200
    assert calls == [expected, expected]
