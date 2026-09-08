"""The idle RPC worker must not retain the previous call's payloads."""

import gc
import time
import weakref

import pytest

from areal.infra.rpc.guard import engine_blueprint


@pytest.mark.parametrize("raises", [False, True])
def test_completed_call_releases_payloads_without_another_call(monkeypatch, raises):
    """Release arguments, closures and Future results on success and failure."""

    class Payload:
        pass

    def submit_payloads():
        positional, keyword, captured = Payload(), Payload(), Payload()
        refs = [weakref.ref(value) for value in (positional, keyword, captured)]

        def call(arg, *, kw):
            if raises:
                raise RuntimeError("test failure")
            return arg, kw, captured

        if raises:
            with pytest.raises(RuntimeError, match="test failure"):
                engine_blueprint._submit_to_engine_thread(
                    "call", call, positional, kw=keyword
                )
        else:
            result = engine_blueprint._submit_to_engine_thread(
                "call", call, positional, kw=keyword
            )
            assert result == (positional, keyword, captured)
        return refs

    monkeypatch.setattr(engine_blueprint, "_engine_thread", None)
    monkeypatch.setattr(engine_blueprint, "_engine_work_queue", None)
    try:
        refs = submit_payloads()
        deadline = time.monotonic() + 2.0
        while any(ref() is not None for ref in refs) and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.01)
        assert all(ref() is None for ref in refs)
        assert engine_blueprint._submit_to_engine_thread("next", lambda: 7) == 7
    finally:
        if engine_blueprint._engine_thread is not None:
            engine_blueprint._engine_work_queue.put(None)
            engine_blueprint._engine_thread.join(timeout=2.0)
            assert not engine_blueprint._engine_thread.is_alive()
