# SPDX-License-Identifier: Apache-2.0

import socket

import pytest

from areal.infra.server_watchdog import InferenceServerWatchdog


@pytest.fixture
def listener():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    yield sock
    sock.close()


def test_listening_server_stays_healthy(listener):
    """Test that a server accepting connections never trips the watchdog."""
    watchdog = InferenceServerWatchdog([listener.getsockname()], max_failures=1)

    for _ in range(3):
        watchdog._probe()

    watchdog.check_health()


def test_refused_connections_raise_after_max_failures(listener):
    """Test that a closed port raises once the failure streak is reached."""
    addr = listener.getsockname()
    watchdog = InferenceServerWatchdog([addr], max_failures=3)
    listener.close()

    watchdog._probe()
    watchdog._probe()
    watchdog.check_health()
    watchdog._probe()

    with pytest.raises(RuntimeError, match=f"{addr[1]} refused 3 consecutive"):
        watchdog.check_health()


def test_successful_probe_resets_the_failure_streak(listener, monkeypatch):
    """Test that only consecutive refusals count."""
    watchdog = InferenceServerWatchdog([listener.getsockname()], max_failures=2)
    real_connect = socket.create_connection
    outcomes = iter([ConnectionRefusedError, None, ConnectionRefusedError])

    def flaky_connect(*args, **kwargs):
        outcome = next(outcomes)
        if outcome is not None:
            raise outcome
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(socket, "create_connection", flaky_connect)

    for _ in range(3):
        watchdog._probe()

    watchdog.check_health()


def test_timeouts_are_not_treated_as_a_dead_server(monkeypatch):
    """Test that a slow or unreachable network does not abort a healthy run."""
    watchdog = InferenceServerWatchdog([("127.0.0.1", 1)], max_failures=1)

    def timeout(*args, **kwargs):
        raise TimeoutError

    monkeypatch.setattr(socket, "create_connection", timeout)

    for _ in range(3):
        watchdog._probe()

    watchdog.check_health()


def test_background_thread_reports_a_dead_server_and_stops(listener):
    """Test that the watch thread detects the failure without manual probes."""
    watchdog = InferenceServerWatchdog(
        [listener.getsockname()], interval=0.01, max_failures=2
    )
    listener.close()

    watchdog.start()
    watchdog._thread.join(timeout=5)

    with pytest.raises(RuntimeError):
        watchdog.check_health()
    watchdog.stop()
