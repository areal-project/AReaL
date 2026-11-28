from collections import Counter

import pytest
import torch
from flask import Flask

from areal.infra.rpc.guard import engine_blueprint
from areal.infra.rpc.guard.app import GuardState
from areal.infra.rpc.guard.engine_blueprint import engine_bp
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.infra.rpc.serialization import serialize_value


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
        ("echo", frozenset(), [False, False], False),
        ("echo", frozenset({"echo"}), [False, False], False),
        ("wait_for_task", frozenset(), [False, False], True),
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
    """Unrelated RPCs preserve aliases only on the grouped rollout output."""

    class FakeEngine:
        def __init__(self):
            self.cpu_staged_rpc_methods = cpu_staged_rpc_methods

        def echo(self, value):
            return value

        def wait_for_task(self, value):
            return value

    localize_calls = []
    remotize_calls = []

    def fake_localize(obj, *, preserve_vision_aliases=False):
        localize_calls.append(preserve_vision_aliases)
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
    assert localize_calls == expected_localize
    assert remotize_calls == [expected_remotize]


@pytest.mark.parametrize(
    ("method", "supports_shared_vision", "is_vision_model", "keyword", "image_fetches"),
    [
        ("compute_logp", True, True, False, 1),
        ("ppo_update", True, True, True, 1),
        ("compute_advantages", True, True, False, 2),
        ("compute_logp", False, True, False, 2),
        ("compute_logp", True, False, False, 2),
    ],
)
def test_call_engine_cold_fetch_shares_only_supported_vision_fields(
    client,
    monkeypatch,
    method,
    supports_shared_vision,
    is_vision_model,
    keyword,
    image_fetches,
):
    """Both vision and token inputs share a fetch batch, with immutable images deduplicated."""
    image = torch.arange(8)
    token = torch.arange(3)
    stored = {"image": image, "other_image": image.clone(), "token": token}
    fetches = []

    class Backend:
        def fetch(self, shards):
            fetches.append([shard.shard_id for shard in shards])
            return [stored[shard.shard_id].clone() for shard in shards]

    class Engine:
        def consume(self, data):
            first, second = data[0]["multi_modal_input"]
            assert (first["pixel_values"] is second["pixel_values"]) == (
                image_fetches == 1
            )
            assert first["pixel_values"] is not second["other"]
            torch.testing.assert_close(second["pixel_values"], image, atol=0, rtol=0)
            assert data[0]["input_ids"] is not data[0]["labels"]
            torch.testing.assert_close(data[0]["input_ids"], token, atol=0, rtol=0)

    def remote(shard):
        return RTensor(TensorShardInfo(shard, "source"), stored[shard].to("meta"))

    engine = Engine()
    engine.is_vision_model = is_vision_model
    if supports_shared_vision:
        engine.shared_vision_rpc_methods = {"compute_logp", "ppo_update"}
    setattr(engine, method, engine.consume)
    monkeypatch.setitem(engine_blueprint._engines, "test", engine)
    monkeypatch.setattr("areal.infra.rpc.rtensor.get_backend", Backend)
    monkeypatch.setattr("areal.infra.rpc.rtensor._fetch_buffer", {})
    monkeypatch.setattr(
        engine_blueprint, "_submit_to_engine_thread", lambda _name, func: func()
    )
    batch = serialize_value(
        [
            {
                "input_ids": remote("token"),
                "labels": remote("token"),
                "multi_modal_input": [
                    {"pixel_values": remote("image")},
                    {"pixel_values": remote("image"), "other": remote("other_image")},
                ],
            }
        ]
    )
    payload = {"kwargs": {"data": batch}} if keyword else {"args": [batch]}
    response = client.post(
        "/call", json={"method": method, "engine_name": "test", **payload}
    )

    assert response.status_code == 200, response.get_json()
    assert len(fetches) == 1
    assert Counter(fetches[0]) == {"image": image_fetches, "other_image": 1, "token": 2}
