# SPDX-License-Identifier: Apache-2.0
"""Delayed readers, cancellation and source admission without CUDA or WeftMoE."""

import asyncio
from threading import Condition
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from areal.infra.sampling_mask import lifetime, service
from areal.infra.sampling_mask.lifetime import LEASE_KEY, LEASES_KEY, LeaseRegistry
from areal.infra.sampling_mask.service import MaskProducer
from areal.infra.sampling_mask.store import RowStore
from areal.utils.sampling_mask import pack_uint24


@pytest.fixture
def ownership(monkeypatch):
    monkeypatch.setattr(lifetime, "gethostip", lambda: "127.0.0.1")
    owner = LeaseRegistry()
    producer = object.__new__(MaskProducer)
    producer.source, producer.nonce = 99, 101
    producer._condition = Condition()
    producer._store = RowStore(1 << 20)
    producer._leases = {}
    producer._next_row = producer._persisted_until = 1000
    producer._fatal_error = None

    def control(source, nonce, path, /, **body):
        if path == "/register":
            return {
                "active": owner.register(
                    body["lease_id"], body["source"], body["source_nonce"]
                )
            }
        assert (source, nonce) == (99, 101)
        assert path == "/release_lease"
        return producer._handle(path, body)

    monkeypatch.setattr(service, "call", control)
    monkeypatch.setattr(lifetime, "call", control)
    yield owner, producer
    owner.close()


def persist(producer, rows, versions, leases):
    n = len(rows)
    producer._persist_rows(
        np.array(rows),
        pack_uint24(torch.tensor([7, 9] * n)),
        torch.arange(n + 1) * 2,
        torch.full((n,), 7),
        torch.tensor(versions),
        leases,
    )


def test_queued_old_version_survives_new_publish_until_all_consumers_finish(ownership):
    """A v166 prefix remains readable after v170 and v200 publication."""
    owner, producer = ownership
    old, new = owner.create(), owner.create()
    persist(producer, [1], [166], [old])
    persist(producer, [2, 3], [170, 200], [new, new])
    owner.release(new[2])
    for _epoch_or_shard in range(4):
        assert producer._lookup([1])[2].tolist() == [166]
        assert producer._store.read([1, 1])[2].tolist() == [7, 7]
    owner.release(old[2])
    owner.release(old[2])  # duplicate acknowledgement
    assert producer._store.nbytes == 0
    assert not producer._leases
    with pytest.raises(ValueError, match="already released"):
        producer._lookup([1])


def test_cancel_before_response_discards_pending_and_future_writes(ownership):
    """Rows need no successfully returned HTTP reference to be reclaimed."""
    owner, producer = ownership
    lease = owner.create()
    persist(producer, [1], [166], [lease])
    owner.release(lease[2])
    persist(producer, [2], [170], [lease])
    assert producer._store.nbytes == 0
    assert not producer._leases
    assert not owner.register(lease[2], 99, 101)


def test_release_between_registration_and_persistence_cannot_resurrect_rows(
    ownership, monkeypatch
):
    """Release wins even when the writer has already passed owner admission."""
    owner, producer = ownership
    lease = owner.create()
    original = service.call

    def race(*args, **kwargs):
        result = original(*args, **kwargs)
        owner.release(lease[2])
        return result

    monkeypatch.setattr(service, "call", race)
    persist(producer, [1], [166], [lease])
    assert producer._store.nbytes == 0
    assert not producer._leases


def test_mixed_sampler_batch_releases_only_finished_workflow(ownership):
    owner, producer = ownership
    first, second = owner.create(), owner.create()
    persist(producer, [1, 2, 3], [1, 1, 1], [first, second, first])
    owner.release(first[2])
    assert producer._lookup([2])[0].tolist() == [2]
    with pytest.raises(ValueError, match="already released"):
        producer._lookup([1])
    owner.release(second[2])
    assert producer._store.nbytes == 0


def test_failed_release_stays_fenced_and_retries(ownership, monkeypatch):
    owner, producer = ownership
    lease = owner.create()
    persist(producer, [1], [166], [lease])
    original = lifetime.call
    monkeypatch.setattr(lifetime, "call", MagicMock(side_effect=TimeoutError))
    with pytest.raises(TimeoutError):
        owner.release(lease[2])
    assert not owner.register(lease[2], 88, 99)
    assert producer._lookup([1])[0].tolist() == [2]
    monkeypatch.setattr(lifetime, "call", original)
    owner.release(lease[2])
    assert not owner._leases
    assert producer._store.nbytes == 0


def test_real_http_owner_rejects_unknown_and_closed_leases(monkeypatch):
    monkeypatch.setattr(lifetime, "gethostip", lambda: "127.0.0.1")
    owner = LeaseRegistry()
    try:
        lease = owner.create()
        lifetime.release_lease(lease)
        for lease_id in [lease[2], lease[2] + 1]:
            result = lifetime.call(
                owner.source,
                owner.nonce,
                "/register",
                lease_id=lease_id,
                source=99,
                source_nonce=101,
            )
            assert result == {"active": False}
        lifetime.release_lease(lease)
        with pytest.raises(RuntimeError, match="generation mismatch"):
            lifetime.call(owner.source, owner.nonce + 1, "/release", lease_id=lease[2])
    finally:
        owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["accept", "reject", "empty", "error", "cancel"])
async def test_workflow_holds_or_releases_whole_lease(outcome, monkeypatch):
    """Includes rows of dropped branches and cancellation without a trajectory."""
    from tests.test_workflow_executor_filtering import _executor, _remote_trajectory

    from areal.infra.workflow_executor import _RolloutTaskInput

    executor, _ = _executor()
    trajectory = _remote_trajectory([0.0, 1.0])
    released = []
    lease = (1, 2, 3)

    class Workflow:
        async def arun_episode(self, engine, data):
            assert LEASE_KEY not in data
            assert lifetime.current_lease.get() == lease
            if outcome == "error":
                raise RuntimeError("generation failed")
            if outcome == "cancel":
                raise asyncio.CancelledError
            return None if outcome == "empty" else trajectory

    async def release(value):
        released.append(value)

    async def clear(traj):
        return None

    monkeypatch.setattr(lifetime, "arelease", release)
    monkeypatch.setattr(executor, "_clear_rejected_trajectory", clear)
    task = executor._create_workflow_task(
        _RolloutTaskInput(
            task_id=1,
            data={LEASE_KEY: lease},
            workflow=Workflow(),
            should_accept_fn=lambda _: outcome != "reject",
        )
    )
    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task()
    else:
        result = await task()
        if outcome == "accept":
            assert result.trajectory[LEASES_KEY] == [lease]
    assert released == ([] if outcome == "accept" else [lease])
    assert lifetime.current_lease.get() is None


def test_rejected_eval_and_duplicate_shard_metadata_release_once(monkeypatch):
    released = []
    monkeypatch.setattr(lifetime, "release_lease", released.append)
    batch = [None, {LEASES_KEY: [[1, 2, 3]]}, {LEASES_KEY: [(1, 2, 3)]}]
    lifetime.release_batch(batch)
    assert released == [(1, 2, 3)]
    assert batch == [None, {}, {}]


def test_registry_shutdown_fences_all_leases_even_when_producer_is_dead(monkeypatch):
    monkeypatch.setattr(lifetime, "gethostip", lambda: "127.0.0.1")
    owner = LeaseRegistry()
    first, second = owner.create(), owner.create()
    owner.register(first[2], 99, 101)
    owner.register(second[2], 98, 101)
    attempted = []

    def fail(source, nonce, path, **body):
        assert not owner.register(second[2], 97, 101)
        attempted.append(source)
        raise TimeoutError("producer unavailable")

    monkeypatch.setattr(lifetime, "call", fail)
    with pytest.raises(ExceptionGroup, match="shutdown failed"):
        owner.close()
    assert set(attempted) == {99, 98}
    assert not owner._thread.is_alive()
    with pytest.raises(RuntimeError, match="registry is closed"):
        owner.create()


@pytest.mark.asyncio
async def test_proxy_session_lease_cannot_be_overridden_by_request(monkeypatch):
    from areal.experimental.openai.proxy import proxy_rollout_server as srv
    from areal.experimental.openai.proxy.server import SessionData

    monkeypatch.setattr(srv, "_openai_client", object())
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setitem(
        srv._session_cache,
        "mask-session",
        SessionData(
            session_id="mask-session",
            metadata={LEASE_KEY: [1, 2, 3]},
        ),
    )

    async def create_fn(metadata=None, **kwargs):
        return metadata

    result = await srv._call_client_create(
        create_fn,
        {"metadata": {LEASE_KEY: [4, 5, 6], "user_key": "value"}},
        "mask-session",
    )
    assert result == {LEASE_KEY: [1, 2, 3], "user_key": "value"}


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel", "accept"])
async def test_controller_owns_lease_before_rpc_and_fences_abandoned_task(
    outcome, monkeypatch
):
    from types import SimpleNamespace

    from tests.test_rollout_controller import create_test_config

    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.infra.controller.rollout_controller import (
        RolloutController,
        _RemoteRolloutTaskInput,
    )

    monkeypatch.setattr(lifetime, "gethostip", lambda: "127.0.0.1")
    config = create_test_config()
    config._sampling_mask_sampling = object()
    controller = RolloutController(RemoteSGLangEngine, config, MagicMock())
    controller._worker_role = "rollout"
    controller._staleness_manager = MagicMock()
    controller._callback_host, controller._callback_port = "127.0.0.1", 1234
    monkeypatch.setattr(
        controller, "_choose_worker", lambda: (SimpleNamespace(id="test"), 0)
    )
    descriptor = None

    async def rpc(worker, method, **kwargs):
        nonlocal descriptor
        if method == "submit":
            descriptor = kwargs["data"][LEASE_KEY]
            assert descriptor[2] in controller._sampling_mask_owner._leases
            if outcome == "timeout":
                raise TimeoutError
            if outcome == "cancel":
                raise asyncio.CancelledError
            controller._pending_futures[1].set_result(None)
            return 1
        return {"trajectory": {LEASES_KEY: [descriptor]}, "metrics": None}

    controller.scheduler.async_call_engine = rpc
    callback = controller._create_submit_callback(
        _RemoteRolloutTaskInput(
            task_id=1,
            data={},
            workflow="unused",
            workflow_kwargs={},
            should_accept_fn=None,
        )
    )
    try:
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await callback()
        else:
            result = await callback()
            if outcome == "accept":
                assert result.trajectory[LEASES_KEY] == [descriptor]
                assert descriptor[2] in controller._sampling_mask_owner._leases
                await asyncio.to_thread(lifetime.release_batch, [result.trajectory])
        assert descriptor is not None
        assert not controller._sampling_mask_owner._leases
    finally:
        controller._sampling_mask_owner.close()
