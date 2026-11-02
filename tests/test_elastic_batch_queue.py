# SPDX-License-Identifier: Apache-2.0

import json
import multiprocessing
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from areal.elastic.infra.state.batch_queue import BatchQueue


def _online_payload(version=0, rows=1):
    import torch

    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]] * rows),
        "attention_mask": torch.ones((rows, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1]] * rows),
        "logprobs": torch.tensor([[0.0, 0.0, -0.3, -0.4]] * rows),
        "versions": torch.tensor([[-1, -1, version, version]] * rows),
        "rewards": torch.ones(rows),
    }


def _receive_episode(inbox, number, version=0, rows=1):
    episode = inbox.admit("source", f"session-{number}", 0)
    inbox.receive(
        episode["episode_id"], episode["lease"], _online_payload(version, rows)
    )
    return episode


@pytest.mark.parametrize("lost_admission", [False, True])
def test_online_explicit_abort_reclaims_credit_once_and_fences_delayed_admit(
    tmp_path, lost_admission
):
    from areal.elastic.infra.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=1, max_staleness=0)
    episode = None if lost_admission else inbox.admit("source", "session", 0)
    kwargs = {
        "episode_id": episode["episode_id"] if episode else None,
        "lease": episode["lease"] if episode else None,
        "reason": "abandoned question",
    }
    cancelled = inbox.cancel_admission("source", "session", 0, **kwargs)
    reopened = OnlineInbox(tmp_path, batch_size=1, max_staleness=0)
    assert reopened.cancel_admission("source", "session", 0, **kwargs) == cancelled
    delayed = reopened.admit("source", "session", 0)
    assert delayed["episode_id"] == cancelled["episode_id"]
    assert delayed["status"] == "rejected"
    assert reopened.status()["active_credits"] == 0
    assert reopened.admit("source", "session", 1)["status"] == "admitted"
    assert reopened.status()["active_credits"] == 1
    with pytest.raises(ValueError, match="lease"):
        reopened.cancel_admission(
            "source",
            "session",
            0,
            episode_id=cancelled["episode_id"],
            lease="wrong",
            reason="retry",
        )
    with pytest.raises(ValueError, match="identity"):
        reopened.cancel_admission(
            "other",
            "session",
            0,
            **{
                **kwargs,
                "episode_id": cancelled["episode_id"],
                "lease": delayed["lease"],
            },
        )


def test_online_abort_admission_race_never_revives_cancelled_credit(tmp_path):
    from areal.elastic.infra.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=1, max_staleness=0)

    def cancel(_):
        return inbox.cancel_admission(
            "source", "session", 0, episode_id=None, lease=None, reason="lost reply"
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda i: cancel(i) if i % 2 else inbox.admit("source", "session", 0),
                range(20),
            )
        )
    assert len({row["episode_id"] for row in results}) == 1
    assert inbox.admit("source", "session", 0)["status"] == "rejected"
    assert inbox.status()["active_credits"] == 0


@pytest.mark.parametrize("reserve", [False, True])
def test_online_abort_cannot_discard_durably_received_episode(tmp_path, reserve):
    from areal.elastic.infra.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=1, max_staleness=0)
    episode = _receive_episode(inbox, 0)
    if reserve:
        assert inbox.seal_batch(0) is not None
    with pytest.raises(ValueError, match="received or checkpoint-bound"):
        inbox.cancel_admission(
            "source",
            "session-0",
            0,
            episode_id=episode["episode_id"],
            lease=episode["lease"],
            reason="too late",
        )
    assert inbox.status()["active_credits"] == 1


def test_online_admission_is_durable_idempotent_and_bounded(tmp_path):
    from areal.elastic.infra.state.episode_inbox import AdmissionFull, OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=4, max_staleness=2)
    first = inbox.admit("source", "session", 0)
    assert inbox.admit("source", "session", 0) == first

    def attempt(number):
        try:
            return inbox.admit("source", str(number), 0)
        except AdmissionFull:
            return None

    with ThreadPoolExecutor(max_workers=4) as executor:
        admitted = [value for value in executor.map(attempt, range(20)) if value]
    assert len(admitted) == 11
    reopened = OnlineInbox(tmp_path, batch_size=4, max_staleness=2)
    assert reopened.status()["active_credits"] == 12
    assert reopened.admit("source", "session", 0) == first
    with pytest.raises(ValueError, match="configuration"):
        OnlineInbox(tmp_path, batch_size=4, max_staleness=0)


def test_online_recovery_gate_is_durable_and_does_not_consume_credit(tmp_path):
    from areal.elastic.infra.state.episode_inbox import AdmissionPaused, OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=4, max_staleness=2)
    first = inbox.admit("source", "first", 0)
    inbox.set_admission_open(False)
    reopened = OnlineInbox(tmp_path, batch_size=4, max_staleness=2)
    for session in ("first", "new"):
        with pytest.raises(AdmissionPaused):
            reopened.admit("source", session, 0)
    assert reopened.status()["active_credits"] == 1
    reopened.set_admission_open(True)
    assert reopened.admit("source", "first", 0) == first


def test_consumer_restores_uncertain_update_without_releasing_its_fence(tmp_path):
    queue = BatchQueue(tmp_path)
    queue.enqueue("episode", policy_version=0, payload={"value": 1})
    consumer = queue.consumer()
    batch = consumer.claim()
    consumer.mark_applied(batch)
    consumer.restore_checkpoint(None)
    with pytest.raises(BlockingIOError):
        queue.consumer()
    replay = consumer.claim()
    assert replay.batch_id == batch.batch_id
    consumer.mark_applied(replay)
    checkpoint = _checkpoint(consumer)
    consumer.restore_checkpoint(checkpoint)
    assert consumer.claim() is None
    consumer.close()


def test_online_receive_survives_lost_ack_and_rejects_conflicting_retry(tmp_path):
    import torch

    from areal.elastic.infra.state.episode_inbox import OnlineInbox
    from areal.infra.rpc.serialization import deserialize_value

    inbox = OnlineInbox(tmp_path, batch_size=4, max_staleness=2)
    episode = _receive_episode(inbox, 0, rows=3)
    inbox = OnlineInbox(tmp_path, batch_size=4, max_staleness=2)
    inbox.receive(episode["episode_id"], episode["lease"], _online_payload(rows=3))
    for _ in range(2):
        inbox.acknowledge_source(episode["episode_id"], episode["lease"])
    assert inbox.status()["active_credits"] == 1
    restored = deserialize_value(
        json.loads(inbox.episode(episode["episode_id"])["payload"])
    )
    torch.testing.assert_close(
        restored["versions"], _online_payload(rows=3)["versions"], rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="different content"):
        inbox.receive(
            episode["episode_id"], episode["lease"], _online_payload(1, rows=3)
        )
    with pytest.raises(ValueError, match="obsolete"):
        inbox.receive(episode["episode_id"], "old-lease", _online_payload())


def test_online_episode_rows_do_not_replace_independent_episode_batch(tmp_path):
    from areal.elastic.infra.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=4, max_staleness=2)
    _receive_episode(inbox, 0, rows=4)
    assert inbox.seal_batch(0) is None
    for number in range(1, 4):
        _receive_episode(inbox, number)
    batch_id = inbox.seal_batch(0)
    assert batch_id is not None
    assert inbox.seal_batch(0) == batch_id
    assert inbox.queue.status()["counts"]["pending"] == 1
    with inbox.queue.consumer() as consumer:
        batch = consumer.claim()
        assert (
            len(batch.payload["trajectories"]) == len(batch.payload["episode_ids"]) == 4
        )
        consumer.mark_applied(batch)
        with pytest.raises(ValueError, match="committed checkpoint"):
            inbox.confirm_publication(1)
        consumer.acknowledge(_checkpoint(consumer))
        assert inbox.status()["active_credits"] == 4
        with pytest.raises(ValueError, match="Reconcile"):
            inbox.seal_batch(1)
        for _ in range(2):
            inbox.confirm_publication(1)
        assert inbox.status()["active_credits"] == 0
        assert consumer.claim() is None


def test_online_checkpoint_recovery_never_replays_committed_update(tmp_path):
    from areal.elastic.infra.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=1, max_staleness=2)
    episode = _receive_episode(inbox, 0)
    batch_id = inbox.seal_batch(0)
    with inbox.queue.consumer() as consumer:
        consumer.mark_applied(consumer.claim())
        checkpoint = _checkpoint(consumer)
    # Latest checkpoint committed; process died before the DB acknowledgement.
    reopened = OnlineInbox(tmp_path, batch_size=1, max_staleness=2)
    with reopened.queue.consumer(checkpoint) as recovered:
        assert recovered.claim() is None
        assert reopened.episode(episode["episode_id"])["batch_id"] == batch_id
        assert reopened.status()["active_credits"] == 1
        reopened.confirm_publication(1)
        assert reopened.status()["active_credits"] == 0


def test_online_staleness_checks_oldest_effective_token_and_preserves_versions(
    tmp_path,
):
    import torch

    from areal.elastic.infra.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=1, max_staleness=2)
    with inbox.queue.consumer() as consumer:
        for version in range(3):
            _receive_episode(inbox, version, version)
            inbox.seal_batch(version)
            consumer.mark_applied(consumer.claim())
            consumer.acknowledge(_checkpoint(consumer))
            inbox.confirm_publication(version + 1)
        episode = inbox.admit("source", "multiturn", 0)
        payload = _online_payload(3)
        payload["versions"][0, 2] = 0  # Last turn alone would look current.
        original = payload["versions"].clone()
        inbox.receive(episode["episode_id"], episode["lease"], payload)
        assert inbox.seal_batch(3) is None
        assert inbox.episode(episode["episode_id"])["rejection"] == "staleness"
        torch.testing.assert_close(payload["versions"], original, rtol=0, atol=0)
        assert inbox.status()["active_credits"] == 0


@pytest.mark.parametrize("kind", ["missing", "meta", "unversioned", "future"])
def test_online_invalid_or_future_training_payload_cannot_be_consumed(tmp_path, kind):
    import torch

    from areal.elastic.infra.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=1, max_staleness=2)
    episode = inbox.admit("source", "session", 0)
    payload = _online_payload()
    if kind == "missing":
        payload.pop("versions")
    elif kind == "meta":
        payload["input_ids"] = torch.empty((1, 4), device="meta")
    elif kind == "unversioned":
        payload["versions"][0, -1] = -1
    else:
        payload["versions"][0, -1] = 1
    if kind == "future":
        inbox.receive(episode["episode_id"], episode["lease"], payload)
        with pytest.raises(ValueError, match="future serving"):
            inbox.seal_batch(0)
    else:
        with pytest.raises(ValueError):
            inbox.receive(episode["episode_id"], episode["lease"], payload)
    assert inbox.queue.status()["counts"]["pending"] == 0


def _checkpoint(consumer):
    return {
        "global_step": consumer.cursor["applied_batches"],
        "data_cursor": consumer.cursor,
    }


def _abrupt_consumer(root, applied):
    queue = BatchQueue(root)
    consumer = queue.consumer()
    consumer.mark_applied(consumer.claim())
    applied.set()
    os._exit(23)


def test_enqueue_duplicate_id_is_idempotent_and_conflict_is_rejected(tmp_path):
    """Producer retries cannot duplicate or silently replace a trajectory batch."""
    queue = BatchQueue(tmp_path)
    first = queue.enqueue("a", {"tokens": [1, 2]}, policy_version=3)
    assert queue.enqueue("a", {"tokens": [1, 2]}, policy_version=3) == first
    for payload, version in (({"tokens": [9]}, 3), ({"tokens": [1, 2]}, 4)):
        with pytest.raises(ValueError, match="different content"):
            queue.enqueue("a", payload, policy_version=version)
    assert queue.status()["counts"]["pending"] == 1


def test_concurrent_producers_preserve_all_batches_and_fifo_order(tmp_path):
    """SQLite serializes independent producers without overwriting IDs."""
    queue = BatchQueue(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        sequences = list(
            pool.map(
                lambda i: queue.enqueue(str(i), {"i": i}, policy_version=0), range(20)
            )
        )
    assert len(set(sequences)) == 20
    with queue.consumer() as consumer:
        seen = []
        while (batch := consumer.claim()) is not None:
            seen.append(batch.sequence)
            consumer.mark_applied(batch)
        assert seen == sorted(sequences)


def test_second_consumer_is_fenced_until_first_closes(tmp_path):
    """A lease never expires while an old trainer can still be updating."""
    queue = BatchQueue(tmp_path)
    first = queue.consumer()
    with pytest.raises(BlockingIOError):
        BatchQueue(tmp_path).consumer()
    first.close()
    with queue.consumer() as second:
        assert second.cursor["applied_batches"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        first.claim()


def test_abrupt_exit_releases_lock_and_replays_uncommitted_batch(tmp_path):
    """OS process death rolls back the cursor even when DB status says applied."""
    queue = BatchQueue(tmp_path)
    queue.enqueue("a", {"value": 1}, policy_version=0)
    context = multiprocessing.get_context("spawn")
    applied = context.Event()
    child = context.Process(target=_abrupt_consumer, args=(tmp_path, applied))
    child.start()
    # Heavy native imports on spawn are setup time, separate from the lock
    # release deadline after the real transaction has executed.
    if not applied.wait(timeout=60):
        child.kill()
        child.join()
        pytest.fail("Consumer process did not apply its transaction")
    child.join(timeout=10)
    if child.is_alive():
        child.kill()
        child.join()
        pytest.fail("Consumer process did not exit")
    assert child.exitcode == 23
    with queue.consumer() as recovered:
        assert recovered.claim().batch_id == "a"


def test_recovery_repairs_missing_ack_and_does_not_replay_committed_prefix(tmp_path):
    """Checkpoint publication commits progress even if queue acknowledgement is lost."""
    queue = BatchQueue(tmp_path)
    for name in ("a", "b", "c"):
        queue.enqueue(name, {"name": name}, policy_version=7)
    with queue.consumer() as consumer:
        consumer.mark_applied(consumer.claim())
        consumer.mark_applied(consumer.claim())
        checkpoint = _checkpoint(consumer)
    assert queue.status()["counts"]["committed"] == 0
    with queue.consumer(checkpoint) as recovered:
        assert queue.status()["counts"]["committed"] == 2
        assert recovered.claim().batch_id == "c"


def test_acknowledgement_cannot_be_rolled_back_to_older_checkpoint(tmp_path):
    """A stale checkpoint must not silently undo already published consumption."""
    queue = BatchQueue(tmp_path)
    queue.enqueue("a", {}, policy_version=0)
    with queue.consumer() as consumer:
        consumer.mark_applied(consumer.claim())
        consumer.acknowledge(_checkpoint(consumer))
    with pytest.raises(ValueError, match="ahead"):
        queue.consumer()


@pytest.mark.parametrize("corruption", ["payload", "missing", "queue_id", "step"])
def test_recovery_rejects_missing_changed_or_foreign_prefix(tmp_path, corruption):
    """A cursor binds batch order and content as well as a particular queue."""
    queue = BatchQueue(tmp_path)
    queue.enqueue("a", {"value": 1}, policy_version=0)
    with queue.consumer() as consumer:
        consumer.mark_applied(consumer.claim())
        checkpoint = _checkpoint(consumer)
    with sqlite3.connect(queue.path) as db:
        if corruption == "payload":
            db.execute("UPDATE batches SET payload='{}'")
        elif corruption == "missing":
            db.execute("DELETE FROM batches")
    if corruption == "queue_id":
        checkpoint["data_cursor"]["queue_id"] = "different"
    if corruption == "step":
        checkpoint["global_step"] = 9
    with pytest.raises(ValueError):
        queue.consumer(checkpoint)


def test_claim_blocks_next_batch_and_checkpoint_until_update_completes(tmp_path):
    """A claimed batch cannot be skipped by snapshotting a later cursor."""
    queue = BatchQueue(tmp_path)
    queue.enqueue("a", {}, policy_version=0)
    with queue.consumer() as consumer:
        batch = consumer.claim()
        with pytest.raises(RuntimeError, match="unapplied"):
            _ = consumer.cursor
        with pytest.raises(RuntimeError, match="not completed"):
            consumer.claim()
        consumer.mark_applied(batch)
        with pytest.raises(ValueError, match="active lease"):
            consumer.mark_applied(batch)
        manifest = _checkpoint(consumer)
        consumer.acknowledge(manifest)
        consumer.acknowledge(manifest)
        assert queue.status()["counts"]["committed"] == 1


def test_claim_rejects_corrupted_pending_payload(tmp_path):
    """Fail before training when stored content no longer matches its digest."""
    queue = BatchQueue(tmp_path)
    queue.enqueue("a", {"value": 1}, policy_version=0)
    with sqlite3.connect(queue.path) as db:
        db.execute("UPDATE batches SET payload='{}'")
    with queue.consumer() as consumer:
        with pytest.raises(ValueError, match="digest"):
            consumer.claim()


def test_peek_validates_without_leasing_or_advancing_the_cursor(tmp_path):
    queue = BatchQueue(tmp_path)
    queue.enqueue("a", {"value": 1}, policy_version=0)
    with queue.consumer() as consumer:
        before = consumer.cursor
        first = consumer.peek()
        assert consumer.peek() == first and first.lease == ""
        assert consumer.cursor == before
        assert queue.status()["counts"]["pending"] == 1
        with pytest.raises(ValueError, match="active lease"):
            consumer.mark_applied(first)
        claimed = consumer.claim()
        assert claimed.batch_id == first.batch_id and claimed.lease
        with pytest.raises(RuntimeError, match="not completed"):
            consumer.peek()
        consumer.mark_applied(claimed)
        assert consumer.peek() is None


def test_peek_rejects_corrupted_payload_before_allocating_a_lease(tmp_path):
    queue = BatchQueue(tmp_path)
    queue.enqueue("a", {"value": 1}, policy_version=0)
    with sqlite3.connect(queue.path) as db:
        db.execute("UPDATE batches SET payload='{}'")
    with queue.consumer() as consumer:
        with pytest.raises(ValueError, match="digest"):
            consumer.peek()
        assert queue.status()["counts"]["pending"] == 1


@pytest.mark.parametrize(
    "payload,version", [({}, -1), ({}, True), ({"x": float("nan")}, 0)]
)
def test_enqueue_invalid_envelope_is_rejected(tmp_path, payload, version):
    """Do not persist ambiguous versions or nonstandard JSON values."""
    with pytest.raises(ValueError):
        BatchQueue(tmp_path).enqueue("a", payload, policy_version=version)
