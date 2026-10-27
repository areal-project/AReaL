import importlib.util
import random
from pathlib import Path

import pytest


def _load_network_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "areal" / "utils" / "network.py"
    spec = importlib.util.spec_from_file_location("areal_utils_network", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_ipv4_only_environment_executes_correct_branch(ip_stack):
    if not (ip_stack["ipv4_route"] and not ip_stack["ipv6_route"]):
        pytest.skip("Not an IPv4-only host (route-level)")
    network = _load_network_module()
    assert ":" not in network.gethostip()


def test_ipv6_only_environment_executes_correct_branch(ip_stack):
    if not (ip_stack["ipv6_route"] and not ip_stack["ipv4_route"]):
        pytest.skip("Not an IPv6-only host (route-level)")
    network = _load_network_module()
    assert ":" in network.gethostip()


def test_dual_stack_environment_executes_correct_branch(ip_stack):
    if not (ip_stack["ipv4_route"] and ip_stack["ipv6_route"]):
        pytest.skip("Not a dual-stack host (route-level)")
    network = _load_network_module()
    ip = network.gethostip()
    assert ip not in {"127.0.0.1", "::1"}


def test_gethostip_uses_custom_ipv6_probe(monkeypatch):
    network = _load_network_module()
    probe_host = "2001:db8::100"
    probe_port = 4321
    connect_calls = []

    def fake_getaddrinfo(host, port, family, socktype):
        if host == "local-host":
            return []
        assert (host, port, family, socktype) == (
            probe_host,
            probe_port,
            network.socket.AF_UNSPEC,
            network.socket.SOCK_DGRAM,
        )
        return [
            (
                network.socket.AF_INET6,
                network.socket.SOCK_DGRAM,
                0,
                "",
                (probe_host, probe_port, 0, 0),
            )
        ]

    class FakeSocket:
        def __init__(self, family, socktype, proto):
            assert family == network.socket.AF_INET6
            assert socktype == network.socket.SOCK_DGRAM
            assert proto == 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def connect(self, sockaddr):
            connect_calls.append(sockaddr)

        def getsockname(self):
            return ("2001:db8::200", 12345, 0, 0)

    monkeypatch.setattr(network.socket, "gethostname", lambda: "local-host")
    monkeypatch.setattr(network.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(network.socket, "socket", FakeSocket)

    assert network.gethostip(probe_host, probe_port) == "2001:db8::200"
    assert connect_calls == [(probe_host, probe_port, 0, 0)]


def test_split_hostport_accepts_unbracketed_ipv6():
    network = _load_network_module()
    host, port = network.split_hostport("2001:db8::1:8000")
    assert host == "2001:db8::1"
    assert port == 8000


def test_split_hostport_accepts_ipv4():
    network = _load_network_module()
    host, port = network.split_hostport("127.0.0.1:8000")
    assert host == "127.0.0.1"
    assert port == 8000


def test_split_hostport_accepts_bracketed_ipv6():
    network = _load_network_module()
    host, port = network.split_hostport("[2001:db8::1]:8000")
    assert host == "2001:db8::1"
    assert port == 8000


def test_split_hostport_accepts_ipv6_with_zone_id_unbracketed():
    network = _load_network_module()
    host, port = network.split_hostport("fe80::1%eth0:8000")
    assert host == "fe80::1%eth0"
    assert port == 8000


def test_split_hostport_accepts_ipv6_with_zone_id_bracketed():
    network = _load_network_module()
    host, port = network.split_hostport("[fe80::1%eth0]:8000")
    assert host == "fe80::1%eth0"
    assert port == 8000


def test_format_hostport_brackets_ipv6():
    network = _load_network_module()
    assert network.format_hostport("2001:db8::1", 8000) == "[2001:db8::1]:8000"


def test_format_host_for_url_idempotent():
    network = _load_network_module()
    assert network.format_host_for_url("[2001:db8::1]") == "[2001:db8::1]"
    assert network.format_host_for_url("127.0.0.1") == "127.0.0.1"


def test_get_loopback_ip_returns_bindable_address():
    network = _load_network_module()
    ip = network.get_loopback_ip()
    assert ip in {"127.0.0.1", "::1"}


def test_find_free_ports_ignores_out_of_range_excludes():
    # Excluded ports outside [min_port, max_port] must not deflate the
    # availability check; all in-range ports here are free.
    network = _load_network_module()
    ports = network.find_free_ports(
        count=6, port_range=(10000, 10005), exclude_ports={50000}
    )
    assert len(ports) == 6


def test_find_free_ports_seeded_candidates_busy_uses_independent_rng(monkeypatch):
    """Port allocation must neither replay nor advance the training RNG."""
    network = _load_network_module()
    training_rng = random.Random(1)
    state = training_rng.getstate()
    occupied = {training_rng.randint(10000, 32767) for _ in range(10)}
    training_rng.setstate(state)
    free_port = next(port for port in range(10000, 32768) if port not in occupied)
    monkeypatch.setattr(network.random, "randint", training_rng.randint)
    monkeypatch.setattr(
        network.random.SystemRandom, "randint", lambda self, lo, hi: free_port
    )
    monkeypatch.setattr(network, "is_port_free", lambda port: port not in occupied)

    assert network.find_free_ports(1) == [free_port]
    assert training_rng.getstate() == state


@pytest.mark.parametrize("occupied", [False, True])
@pytest.mark.parametrize("start_port", [10000, 10010, 10020])
def test_find_free_ports_sparse_availability_returns_remaining_port(
    monkeypatch, occupied, start_port
):
    """Find a sole eligible port regardless of the random starting point."""
    network = _load_network_module()
    free_port = 10010
    unavailable = set(range(10000, 10021)) - {free_port}
    checked = []

    def is_port_free(port):
        checked.append(port)
        return port == free_port

    monkeypatch.setattr(
        network.random.SystemRandom, "randint", lambda self, lo, hi: start_port
    )
    monkeypatch.setattr(network, "is_port_free", is_port_free)

    assert network.find_free_ports(
        1, (10000, 10020), exclude_ports=set() if occupied else unavailable
    ) == [free_port]
    assert len(checked) == len(set(checked))
    if not occupied:
        assert checked == [free_port]


def test_find_free_ports_full_range_returns_sorted_unique_ports(monkeypatch):
    """Repeated random draws must not prevent filling an entirely free range."""
    network = _load_network_module()
    checked = []
    monkeypatch.setattr(
        network.random.SystemRandom, "randint", lambda self, lo, hi: 10003
    )
    monkeypatch.setattr(
        network, "is_port_free", lambda port: checked.append(port) or True
    )

    assert network.find_free_ports(6, (10000, 10005)) == list(range(10000, 10006))
    assert len(checked) == len(set(checked)) == 6


def test_find_free_ports_exhausted_range_checks_all_eligible_ports_once(monkeypatch):
    """Report insufficient free ports only after inspecting every candidate."""
    network = _load_network_module()
    excluded = {10001, 10004}
    checked = []

    def is_port_free(port):
        checked.append(port)
        return port == 10003

    monkeypatch.setattr(
        network.random.SystemRandom, "randint", lambda self, lo, hi: 10003
    )
    monkeypatch.setattr(network, "is_port_free", is_port_free)

    with pytest.raises(ValueError, match="Could only find 1 free ports out of 2"):
        network.find_free_ports(2, (10000, 10005), exclude_ports=excluded)

    assert set(checked) == set(range(10000, 10006)) - excluded
    assert len(checked) == len(set(checked))
