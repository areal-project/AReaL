# SPDX-License-Identifier: Apache-2.0

"""Tests for ``areal weight-update`` (CPU-only, no service processes).

The gateway is stubbed at the HTTP-client boundary rather than launched: these
verbs are pure read paths, and what is worth pinning is how they behave when
the state file is missing, unreadable, or points at something that is not
answering -- none of which a live gateway would reproduce on demand.

Test naming convention: test_<what>_<condition>_<expected>()
"""

import json

import pytest

from areal.v2.cli.client import ServiceUnreachable
from areal.v2.cli.weight_update import weight_update
from areal.v2.cli.weight_update.commands import pairs as pairs_mod
from areal.v2.cli.weight_update.commands import ps as ps_mod
from areal.v2.cli.weight_update.commands import status as status_mod
from areal.v2.cli.weight_update.state import GatewayHandle, ServiceState, store


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own ~/.areal. The store re-resolves paths on each
    call, so setting the env var is enough -- no store rebuild needed."""
    monkeypatch.setenv("AREAL_HOME", str(tmp_path))
    yield


def _write_state(service="default", url="http://127.0.0.1:7080"):
    state = ServiceState(
        service=service,
        launch_mode="controller",
        admin_api_key="k",
        gateway=GatewayHandle(url=url, pid=4242),
    )
    state.save()
    return state


class TestState:
    def test_state_round_trips_through_disk(self):
        written = _write_state()
        loaded = ServiceState.load("default")
        assert loaded.gateway.url == written.gateway.url
        assert loaded.gateway.pid == 4242
        assert loaded.launch_mode == "controller"

    def test_gateway_handle_exposes_addr_for_component_probe(self):
        # ServiceStateBase.components() yields handles that scaffold probes
        # structurally; it needs .addr and .pid, not a base class.
        handle = GatewayHandle(url="http://h:1", pid=7)
        assert handle.addr == "http://h:1" and handle.pid == 7

    def test_remove_deletes_the_state_file(self):
        _write_state()
        assert store.service_state_path("default").exists()
        ServiceState.remove("default")
        assert not store.service_state_path("default").exists()


class TestResolveTarget:
    def test_no_state_and_no_gateway_flag_names_the_escape_hatch(self):
        import click

        with pytest.raises(click.ClickException) as err:
            status_mod.do_status(False)
        assert "--gateway" in str(err.value)

    def test_gateway_flag_wins_over_local_state(self, monkeypatch):
        _write_state(url="http://from-state:7080")
        seen = {}

        def fake_health(self, **kw):
            seen["base"] = self.base
            return {}

        monkeypatch.setattr(
            "areal.v2.cli.weight_update.client.WeightUpdateClient.health", fake_health
        )
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.client.WeightUpdateClient.pairs",
            lambda self, **kw: [],
        )
        status_mod.do_status(True, gateway="http://explicit:9000")
        assert seen["base"] == "http://explicit:9000"


class TestStatus:
    def test_unreachable_gateway_exits_non_zero(self, monkeypatch, capsys):
        _write_state()
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.client.WeightUpdateClient.health",
            lambda self, **kw: (_ for _ in ()).throw(ServiceUnreachable("refused")),
        )
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.client.WeightUpdateClient.pairs",
            lambda self, **kw: (_ for _ in ()).throw(ServiceUnreachable("refused")),
        )
        # Non-zero so the verb can gate a script, not just print a word.
        assert status_mod.do_status(True) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["reachable"] is False and payload["pairs"] is None

    def test_reachable_gateway_reports_pair_count_and_exits_zero(
        self, monkeypatch, capsys
    ):
        _write_state()
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.client.WeightUpdateClient.health",
            lambda self, **kw: {},
        )
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.client.WeightUpdateClient.pairs",
            lambda self, **kw: [{"pair_name": "a"}, {"pair_name": "b"}],
        )
        assert status_mod.do_status(True) == 0
        assert json.loads(capsys.readouterr().out)["pairs"] == 2


class TestPairs:
    def test_pairs_renders_the_fields_an_operator_acts_on(self, monkeypatch, capsys):
        _write_state()
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.client.WeightUpdateClient.pairs",
            lambda self, **kw: [
                {
                    "pair_name": "actor",
                    "mode": "awex",
                    "train_world_size": 4,
                    "inference_world_size": 2,
                    "last_version": 17,
                    "colocate": True,
                }
            ],
        )
        assert pairs_mod.do_pairs(False) == 0
        out = capsys.readouterr().out
        for token in ("actor", "awex", "17", "yes"):
            assert token in out

    def test_no_pairs_says_so_rather_than_printing_an_empty_table(
        self, monkeypatch, capsys
    ):
        _write_state()
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.client.WeightUpdateClient.pairs",
            lambda self, **kw: [],
        )
        pairs_mod.do_pairs(False)
        assert "no pairs connected" in capsys.readouterr().out


class TestPs:
    def test_ps_with_no_services_says_so(self, capsys):
        assert ps_mod.do_ps(False) == 0
        assert "no weight-update services" in capsys.readouterr().out

    def test_ps_marks_a_gateway_that_is_not_answering(self, monkeypatch, capsys):
        _write_state()
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.state.ServiceState.gateway_alive",
            lambda self: False,
        )
        ps_mod.do_ps(False, True)
        assert "unreachable" in capsys.readouterr().out

    def test_a_dead_service_is_hidden_until_all_is_passed(self, monkeypatch, capsys):
        # Same default as `areal inf ps` and `areal agent ps`.
        _write_state(service="dead")
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.state.ServiceState.gateway_alive",
            lambda self: False,
        )
        ps_mod.do_ps(False, False)
        assert "dead" not in capsys.readouterr().out
        ps_mod.do_ps(False, True)
        assert "dead" in capsys.readouterr().out

    def test_unreadable_state_file_does_not_hide_the_others(self, monkeypatch, capsys):
        _write_state(service="good")
        bad = store.service_state_path("broken")
        bad.write_text("{ not json")
        monkeypatch.setattr(
            "areal.v2.cli.weight_update.state.ServiceState.gateway_alive",
            lambda self: True,
        )
        ps_mod.do_ps(False, True)
        out = capsys.readouterr().out
        assert "good" in out and "unreadable" in out


class TestParser:
    def test_group_exposes_exactly_the_first_cut_verbs(self):
        # `logs` is deliberately absent: the controller inherits the gateway's
        # stdout instead of redirecting it to a file, so adding the verb means
        # changing where that output goes -- a separate change.
        assert set(weight_update.commands) == {"status", "pairs", "ps"}

    def test_the_installed_console_script_exposes_the_group(self):
        # `areal` binds areal.v2.cli.main:cli. Importing the group directly
        # stays green when the add_command line is deleted; this does not.
        from areal.v2.cli.main import cli as console_entry

        assert "weight-update" in console_entry.commands

    def test_ps_matches_the_flags_of_the_sibling_namespaces(self):
        from areal.v2.cli.inference.commands.ps import ps_cmd as inf_ps

        mine = {o for p in ps_mod.ps_cmd.params for o in p.opts}
        theirs = {o for p in inf_ps.params for o in p.opts}
        assert mine == theirs, f"`weight-update ps` flags drifted: {mine ^ theirs}"

    def test_every_verb_offers_json_output(self):
        for name, cmd in weight_update.commands.items():
            assert any("--json" in p.opts for p in cmd.params), (
                f"{name} has no --json flag"
            )
