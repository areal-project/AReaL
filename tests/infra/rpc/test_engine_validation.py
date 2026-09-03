import pytest
from flask import Flask

from areal.infra.rpc.guard import engine_blueprint
from areal.infra.rpc.guard.app import GuardState
from areal.infra.rpc.guard.engine_blueprint import engine_bp


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
