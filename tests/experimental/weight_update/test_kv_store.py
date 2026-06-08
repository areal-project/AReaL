# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

import pytest

from areal.experimental.weight_update.gateway.kv_store import WeightMetaStore


@pytest.fixture()
def store() -> WeightMetaStore:
    return WeightMetaStore()


def test_put_get_roundtrip_simple_dict(store: WeightMetaStore):
    payload = {"version": 3, "status": "ready"}
    store.put("pair_a", "meta", payload)
    assert store.get("pair_a", "meta") == payload


def test_put_get_roundtrip_nested_complex(store: WeightMetaStore):
    payload = {
        "weights": [1.0, 2.0, 3.0],
        "config": {"lr": 0.001, "layers": [64, 128]},
        "tags": None,
    }
    store.put("pair_a", "complex", payload)
    assert store.get("pair_a", "complex") == payload


def test_get_missing_key_returns_none(store: WeightMetaStore):
    assert store.get("nonexistent_pair", "key") is None
    store.put("pair_a", "k1", "v1")
    assert store.get("pair_a", "missing") is None


def test_delete_removes_entry(store: WeightMetaStore):
    store.put("pair_a", "k1", "v1")
    assert store.delete("pair_a", "k1") is True
    assert store.get("pair_a", "k1") is None


def test_delete_nonexistent_returns_false(store: WeightMetaStore):
    assert store.delete("pair_a", "nope") is False


def test_add_to_set_and_set_size(store: WeightMetaStore):
    store.add_to_set("pair_a", "barrier_1", "worker_0")
    store.add_to_set("pair_a", "barrier_1", "worker_1")
    assert store.set_size("pair_a", "barrier_1") == 2
    # Duplicate add should not increase size
    store.add_to_set("pair_a", "barrier_1", "worker_0")
    assert store.set_size("pair_a", "barrier_1") == 2


def test_set_size_nonexistent_returns_zero(store: WeightMetaStore):
    assert store.set_size("pair_a", "no_set") == 0


def test_clear_pair_removes_all_data(store: WeightMetaStore):
    store.put("pair_a", "k1", "v1")
    store.put("pair_a", "k2", "v2")
    store.add_to_set("pair_a", "barrier", "w0")
    store.clear_pair("pair_a")
    assert store.get("pair_a", "k1") is None
    assert store.get("pair_a", "k2") is None
    assert store.set_size("pair_a", "barrier") == 0
    assert store.list_keys("pair_a") == []


def test_pair_isolation(store: WeightMetaStore):
    store.put("pair_a", "key", "value_a")
    store.put("pair_b", "key", "value_b")
    assert store.get("pair_a", "key") == "value_a"
    assert store.get("pair_b", "key") == "value_b"
    store.clear_pair("pair_a")
    assert store.get("pair_a", "key") is None
    assert store.get("pair_b", "key") == "value_b"


def test_list_keys(store: WeightMetaStore):
    store.put("pair_a", "k1", 1)
    store.put("pair_a", "k2", 2)
    keys = store.list_keys("pair_a")
    assert sorted(keys) == ["k1", "k2"]
    assert store.list_keys("empty_pair") == []


def test_thread_safety_concurrent_puts(store: WeightMetaStore):
    num_threads = 10
    writes_per_thread = 100
    errors: list[Exception] = []

    def writer(thread_id: int) -> None:
        try:
            for i in range(writes_per_thread):
                store.put("pair_a", f"t{thread_id}_k{i}", f"v{thread_id}_{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    total_keys = len(store.list_keys("pair_a"))
    assert total_keys == num_threads * writes_per_thread
