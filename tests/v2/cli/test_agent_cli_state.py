# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from areal.v2.cli.agent.config import load_click_default_map
from areal.v2.cli.agent.state import (
    ProcessState,
    ServiceState,
    store,
)


def test_service_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    state = ServiceState(
        service="svc",
        launch_mode="detached",
        agent="pkg.Agent",
        admin_api_key="key",
        gateway=ProcessState(
            component="gateway",
            pid=11,
            url="http://127.0.0.1:1",
            log_file="gateway.log",
        ),
        router=ProcessState(
            component="router",
            pid=12,
            url="http://127.0.0.1:2",
            log_file="router.log",
        ),
        pairs=[],
    )

    state.save()

    loaded = ServiceState.load("svc")
    assert loaded.service == "svc"
    assert loaded.gateway.pid == 11
    assert [pid for _, h in loaded.components() if (pid := h.pid) > 0] == [11, 12]
    assert store.service_state_path("svc").exists()


def test_config_maps_toml_sections_to_click_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_HOME", str(tmp_path / "home"))
    config_dir = tmp_path / "home" / "agent"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        "[default]\nadmin_api_key = 'base'\n[run]\nagent = 'pkg.Agent'\nnum_pairs = 4\n"
    )
    extra = tmp_path / "extra.toml"
    extra.write_text("[default]\nadmin_api_key = 'override'\n")

    default_map = load_click_default_map(extra=extra)

    assert default_map["run"]["admin_api_key"] == "override"
    assert default_map["run"]["agent"] == "pkg.Agent"
    assert default_map["run"]["num_pairs"] == 4
