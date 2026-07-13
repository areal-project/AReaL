import pytest
from flask import Flask

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
