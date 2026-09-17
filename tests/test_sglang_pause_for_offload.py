from types import SimpleNamespace

from areal.engine.sglang_remote import SGLangBackend
from areal.infra.remote_inf_engine import RemoteInfEngine


def test_pause_generation_drains_then_pauses_scheduler():
    requests = []
    engine = RemoteInfEngine.__new__(RemoteInfEngine)
    engine.backend = SGLangBackend()
    engine.config = SimpleNamespace(pause_grace_period=0)
    engine._run_request_on_all_servers = requests.append

    pause = getattr(
        RemoteInfEngine.pause_generation,
        "__wrapped__",
        RemoteInfEngine.pause_generation,
    )
    pause(engine)

    assert [request.payload for request in requests] == [
        {},
        {"mode": "in_place"},
    ]


def test_default_pause_request_matches_upstream_payload():
    assert SGLangBackend().get_pause_request().payload == {}


def test_pause_generation_keeps_single_request_backends_compatible():
    requests = []

    class _SingleRequestBackend:
        def get_pause_request(self):
            return SimpleNamespace(payload={"backend": "single"})

    engine = RemoteInfEngine.__new__(RemoteInfEngine)
    engine.backend = _SingleRequestBackend()
    engine.config = SimpleNamespace(pause_grace_period=0)
    engine._run_request_on_all_servers = requests.append

    pause = getattr(
        RemoteInfEngine.pause_generation,
        "__wrapped__",
        RemoteInfEngine.pause_generation,
    )
    pause(engine)

    assert [request.payload for request in requests] == [{"backend": "single"}]
