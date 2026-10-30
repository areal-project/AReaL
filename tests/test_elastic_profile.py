# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

from tests.elastic_profile import (
    OSProxy,
    Recorder,
    hook,
    spans_in_window,
    union_seconds,
)


def test_parallel_spans_use_union_not_sum():
    assert (
        union_seconds(
            [
                (0, 4_000_000_000),
                (1_000_000_000, 3_000_000_000),
                (3_000_000_000, 6_000_000_000),
                (8_000_000_000, 9_000_000_000),
            ]
        )
        == 7
    )


def test_measured_window_excludes_identical_audit_hooks():
    rows = [
        dict(name="state_dict", start_ns=start, end_ns=end)
        for start, end in ((0, 10), (15, 25), (29, 33), (35, 40))
    ]
    selected = spans_in_window(rows, dict(start_ns=12, end_ns=30))
    assert [(r["start_ns"], r["end_ns"]) for r in selected] == [(15, 25), (29, 30)]


@pytest.mark.asyncio
async def test_hooks_preserve_results_exceptions_and_hide_arguments(tmp_path):
    async def rpc(secret):
        if secret == "bad":
            raise ValueError("secret error content")
        return secret

    target = SimpleNamespace(rpc=rpc)
    recorder = Recorder(tmp_path, "unit")
    with ExitStack() as stack:
        hook(stack, target, "rpc", recorder, "rpc.fixed_label")
        assert await target.rpc("private token") == "private token"
        with pytest.raises(ValueError, match="secret error content"):
            await target.rpc("bad")
    assert target.rpc is rpc
    text = next(tmp_path.glob("*.jsonl")).read_text()
    assert "private token" not in text and "secret error content" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert [row["status"] for row in rows] == ["ok", "ValueError"]


def test_os_proxy_does_not_change_global_fsync(tmp_path):
    import os

    original = os.fsync
    proxy = OSProxy(os, Recorder(tmp_path, "unit"), "store")
    with (tmp_path / "data").open("wb") as stream:
        stream.write(b"hello")
        stream.flush()
        proxy.fsync(stream.fileno())
    assert os.fsync is original
    row = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
    assert row["name"] == "store.fsync_file"


def test_profile_actor_forbids_training_entry_points():
    import importlib.util

    if importlib.util.find_spec("megatron") is None:
        pytest.skip("Requires the installed Megatron/TransformerEngine environment")
    from tests.elastic_profile_actor import ProfileActor

    for name in (
        "train_batch",
        "forward_batch",
        "forward_backward_batch",
        "eval_batch",
        "optimizer_step",
    ):
        with pytest.raises(RuntimeError, match="prohibited"):
            getattr(ProfileActor, name)(None)
