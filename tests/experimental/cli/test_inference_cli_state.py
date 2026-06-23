# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

from click.testing import CliRunner

from areal.experimental.cli.inference.common import resolve_model_name
from areal.experimental.cli.inference.state import (
    ModelEntry,
    ModelState,
    ServiceState,
    current_service_path,
    models_state_path,
    recover_pids_from_raw_state,
    resolve_service_name,
    service_state_path,
)
from areal.experimental.cli.main import cli


def _save_service(service: str, *, gateway_pid: int | None = None) -> None:
    pid = os.getpid() if gateway_pid is None else gateway_pid
    ServiceState(
        service=service,
        gateway_pid=pid,
        gateway_url=f"http://127.0.0.1/{service}",
        router_pid=pid,
        router_url=f"http://127.0.0.1/{service}-router",
        admin_api_key="admin",
        started_at=1.0,
    ).save()


def test_service_and_model_state_are_per_service(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("svc-a")
    model_state = ModelState(service="svc-a")
    model_state.models["m-a"] = ModelEntry(kind="external", api_url="http://api")
    model_state.set_default_if_empty("m-a")
    model_state.save()

    assert service_state_path("svc-a").exists()
    assert models_state_path("svc-a").exists()
    assert current_service_path().read_text().strip() == "svc-a"

    loaded_service = ServiceState.load("svc-a")
    loaded_models = ModelState.load("svc-a")
    assert loaded_service.service == "svc-a"
    assert loaded_models.default_model == "m-a"
    assert list(loaded_models.models) == ["m-a"]


def test_resolve_service_uses_current_then_single_running(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("current")
    assert resolve_service_name(None) == "current"

    current_service_path().unlink()
    assert resolve_service_name(None) == "current"


def test_models_command_is_service_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("svc-a")
    _save_service("svc-b")
    for service, model in (("svc-a", "m-a"), ("svc-b", "m-b")):
        model_state = ModelState(service=service)
        model_state.models[model] = ModelEntry(
            kind="external", api_url=f"http://{model}"
        )
        model_state.set_default_if_empty(model)
        model_state.save()

    result = CliRunner().invoke(cli, ["inf", "models", "--service", "svc-a", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [entry["name"] for entry in payload] == ["m-a"]


def test_ps_lists_services_and_stale_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("live")
    _save_service("stale", gateway_pid=-1)

    result = CliRunner().invoke(cli, ["inf", "ps", "--all", "--json"])

    assert result.exit_code == 0
    payload = {entry["service"]: entry["status"] for entry in json.loads(result.output)}
    assert payload == {"live": "running", "stale": "stale"}


def test_default_model_resolution_uses_model_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("svc")
    model_state = ModelState(service="svc")
    model_state.models["m"] = ModelEntry(kind="external")
    model_state.set_default_if_empty("m")
    model_state.save()

    from areal.experimental.cli.inference.common import load_running_state

    state = load_running_state("svc")

    assert resolve_model_name(state, None) == "m"
    assert resolve_model_name(state, "explicit") == "explicit"


def test_model_entry_splits_legacy_interleaved_pids():
    entry = ModelEntry(pids=[10, 20, 11, 21])

    assert entry.engine_pids == [10, 11]
    assert entry.proxy_pids == [20, 21]
    assert entry.all_pids() == [10, 20, 11, 21]


def test_recover_pids_from_raw_state(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("svc", gateway_pid=100)
    service_payload = json.loads(service_state_path("svc").read_text())
    service_payload["router_pid"] = 101
    service_state_path("svc").write_text(json.dumps(service_payload))
    models_state_path("svc").write_text(
        json.dumps(
            {
                "models": {
                    "m": {
                        "engine_pids": [200],
                        "proxy_pids": [300],
                        "pids": [200, 300],
                    }
                }
            }
        )
    )

    assert recover_pids_from_raw_state("svc") == [100, 101, 200, 300]
