# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

from click.testing import CliRunner

from areal.v2.cli.inference.scheduler import TaskHandle
from areal.v2.cli.inference.state import (
    ModelEntry,
    ModelState,
    ServiceState,
    store,
)
from areal.v2.cli.main import cli


def _service_path(service: str):
    return store.service_state_path(service)


def _current_path():
    return store.current_service_path()


def _resolve(name: str | None) -> str:
    return store.resolve_service_name(name)


def _save_service(service: str, *, gateway_pid: int | None = None) -> None:
    pid = os.getpid() if gateway_pid is None else gateway_pid
    ServiceState(
        service=service,
        backend="local",
        gateway_handle=TaskHandle(
            host="127.0.0.1", ports=[8080], gpu_devices=[], ref={"pid": pid}
        ),
        router_handle=TaskHandle(
            host="127.0.0.1", ports=[9000], gpu_devices=[], ref={"pid": pid}
        ),
        admin_api_key="admin",
        started_at=1.0,
    ).save()


def _placeholder_model() -> ModelEntry:
    return ModelEntry(backend="sglang:d1", replicas=[])


def test_service_and_model_state_are_per_service(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("svc-a")
    model_state = ModelState(service="svc-a")
    model_state.models["m-a"] = _placeholder_model()
    model_state.save()

    assert _service_path("svc-a").exists()
    assert store.models_state_path("svc-a").exists()
    assert _current_path().read_text().strip() == "svc-a"

    loaded_service = ServiceState.load("svc-a")
    loaded_models = ModelState.load("svc-a")
    assert loaded_service.service == "svc-a"
    assert loaded_service.backend == "local"
    assert list(loaded_models.models) == ["m-a"]


def test_resolve_service_uses_current_then_single_running(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("current")
    assert _resolve(None) == "current"

    _current_path().unlink()
    assert _resolve(None) == "current"


def test_models_command_is_service_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    _save_service("svc-a")
    _save_service("svc-b")
    for service, model in (("svc-a", "m-a"), ("svc-b", "m-b")):
        model_state = ModelState(service=service)
        model_state.models[model] = _placeholder_model()
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


def test_recover_pids_from_raw_state_walks_handles(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    # Hand-write the v1 schema so we exercise recover_pids_from_raw_state
    # against TaskHandle.ref.pid nesting. Order of returned pids follows the
    # JSON walk order: service file first, then models file; within each
    # ModelReplica data_proxy is declared before worker.
    _service_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "backend": "local",
                "gateway_handle": {
                    "host": "127.0.0.1",
                    "ports": [8080],
                    "gpu_devices": [],
                    "ref": {"pid": 100},
                },
                "router_handle": {
                    "host": "127.0.0.1",
                    "ports": [9000],
                    "gpu_devices": [],
                    "ref": {"pid": 101},
                },
                "admin_api_key": "admin",
                "started_at": 1.0,
            }
        )
    )
    store.models_state_path("svc").write_text(
        json.dumps(
            {
                "service": "svc",
                "models": {
                    "m": {
                        "backend": "sglang:d1",
                        "replicas": [
                            {
                                "data_proxy": {
                                    "host": "127.0.0.1",
                                    "ports": [5001],
                                    "gpu_devices": [],
                                    "ref": {"pid": 300},
                                },
                                "worker": {
                                    "host": "127.0.0.1",
                                    "ports": [5000],
                                    "gpu_devices": [0],
                                    "ref": {"pid": 200},
                                },
                            }
                        ],
                    }
                },
            }
        )
    )

    assert store.recover_pids_from_raw_state("svc") == [100, 101, 300, 200]
