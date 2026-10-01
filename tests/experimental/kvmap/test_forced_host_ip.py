# SPDX-License-Identifier: Apache-2.0
"""AREAL_FORCE_HOST_IP makes every service bind and advertise one address; without it nothing changes."""

import pytest

from areal.utils import network


def test_forced_address_is_used_for_binding_and_advertising(monkeypatch):
    monkeypatch.setenv(network.FORCE_HOST_IP_ENV, "127.0.0.1")
    assert network.forced_host_ip() == "127.0.0.1"
    assert network.default_bind_host() == "127.0.0.1"
    assert network.gethostip() == "127.0.0.1"


@pytest.mark.parametrize("value", [None, ""])
def test_unset_or_empty_keeps_all_interfaces(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(network.FORCE_HOST_IP_ENV, raising=False)
    else:
        monkeypatch.setenv(network.FORCE_HOST_IP_ENV, value)
    assert network.forced_host_ip() is None
    assert network.default_bind_host() == "0.0.0.0"


def test_malformed_address_is_rejected(monkeypatch):
    monkeypatch.setenv(network.FORCE_HOST_IP_ENV, "localhost:80")
    with pytest.raises(ValueError):
        network.forced_host_ip()
