"""Worker-scoped Arena registration cleanup regressions."""

from types import SimpleNamespace

from areal.api.cli_args import InferenceEngineConfig


def test_remote_inf_engine_destroy_runs_each_registered_callback_once() -> None:
    from areal.infra.remote_inf_engine import RemoteInfEngine

    engine = RemoteInfEngine(
        InferenceEngineConfig(setup_timeout=60),
        backend=SimpleNamespace(),
    )
    calls = []
    engine.register_destroy_callback("arena", lambda: calls.append("first"))
    engine.register_destroy_callback("arena", lambda: calls.append("duplicate"))

    engine.destroy()
    engine.destroy()

    assert calls == ["first"]


def test_remote_inf_engine_destroy_callback_failure_does_not_mask_cleanup() -> None:
    from areal.infra.remote_inf_engine import RemoteInfEngine

    engine = RemoteInfEngine(
        InferenceEngineConfig(setup_timeout=60),
        backend=SimpleNamespace(),
    )
    calls = []

    def fail() -> None:
        calls.append("fail")
        raise RuntimeError("cleanup failed")

    engine.register_destroy_callback("later", lambda: calls.append("later"))
    engine.register_destroy_callback("fail", fail)

    engine.destroy()

    assert calls == ["fail", "later"]
