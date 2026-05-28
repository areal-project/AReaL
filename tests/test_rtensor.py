"""Integration tests for RTensor with RPC server."""

import asyncio
import subprocess
import sys
import time
import uuid

import orjson
import pytest
import requests
import torch

from areal.infra.rpc.rtensor import (
    HttpRTensorBackend,
    RTensor,
    TensorShardInfo,
)
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.utils.proc import kill_process_tree
from areal.utils.network import find_free_ports


@pytest.fixture(scope="module")
def rpc_server():
    """Start RPC server for integration tests."""
    RPC_SERVER_PORT = find_free_ports(1)[0]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "areal.infra.rpc.rpc_server",
            "--host",
            "localhost",
            "--port",
            str(RPC_SERVER_PORT),
            "--experiment-name",
            "test-rtensor",
            "--trial-name",
            "trial0",
            "--role",
            "master",
            "--worker-index",
            "0",
        ],
        stdout=sys.stdout,
        stderr=sys.stdout,
    )

    # Wait for server to be ready
    max_attempts = 60
    for _ in range(max_attempts):
        try:
            resp = requests.get(f"http://localhost:{RPC_SERVER_PORT}/health", timeout=1)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError("RPC server failed to start")

    yield f"localhost:{RPC_SERVER_PORT}"

    kill_process_tree(proc.pid)


class TestRTensorIntegration:
    """Integration tests using real RPC server."""

    def test_single_shard_storage_and_retrieval(self, rpc_server):
        """Test storing and retrieving a single tensor shard (InferenceEngine workflow)."""
        # Create tensor and shard ID
        tensor = torch.randn(5, 10).cpu()
        shard_id = str(uuid.uuid4())

        # Create RTensor manually
        rtensor = RTensor(
            shard=TensorShardInfo(
                shard_id=shard_id,
                node_addr=rpc_server,
            ),
            data=tensor.to("meta"),
        )

        # Verify RTensor structure
        assert rtensor.shard.shard_id == shard_id
        assert rtensor.shard.node_addr == rpc_server
        assert rtensor.shape[0] == tensor.shape[0]

        # Store on server
        serialized_tensor = serialize_value(tensor)
        resp = requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized_tensor),
        )
        assert resp.status_code == 200

        # Retrieve via RTensor.to_local()
        localized = rtensor.to_local()

        assert isinstance(localized, torch.Tensor)
        assert localized.shape == tensor.shape
        assert torch.allclose(localized, tensor)

    def test_localize_nested_structure(self, rpc_server):
        """Test localizing nested structures containing RTensors."""
        # Create tensors
        tensor1 = torch.randn(3, 4).cpu()
        tensor2 = torch.randn(2, 6).cpu()

        # Store on server
        shard_id1 = str(uuid.uuid4())
        shard_id2 = str(uuid.uuid4())

        for shard_id, tensor in [(shard_id1, tensor1), (shard_id2, tensor2)]:
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        # Create nested structure with RTensors
        nested = {
            "logits": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id1,
                    node_addr=rpc_server,
                ),
                data=torch.empty(tensor1.shape, device="meta"),
            ),
            "metadata": {"count": 3},
            "values": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id2,
                    node_addr=rpc_server,
                ),
                data=torch.empty(tensor2.shape, device="meta"),
            ),
        }

        # Localize entire structure
        localized = RTensor.localize(nested)

        # Verify structure
        assert isinstance(localized["logits"], torch.Tensor)
        assert isinstance(localized["values"], torch.Tensor)
        assert localized["metadata"]["count"] == 3
        assert torch.allclose(localized["logits"], tensor1)
        assert torch.allclose(localized["values"], tensor2)

    def test_remotize_and_localize_roundtrip(self, rpc_server):
        """Test remotize and localize roundtrip."""
        # Simulate output with tensors
        output = {
            "logits": torch.randn(4, 10).cpu(),
            "score": 0.95,
        }

        # remotize using new 2-arg signature
        remotized = RTensor.remotize(output, node_addr=rpc_server)

        # Verify RTensor was created
        assert isinstance(remotized["logits"], RTensor)
        assert remotized["score"] == 0.95

        # Store tensor on server using the NEW shard_id created by remotize
        from areal.infra.rpc.rtensor import fetch

        actual_shard_id = remotized["logits"].shard.shard_id
        tensor_from_local = fetch(actual_shard_id)
        serialized = serialize_value(tensor_from_local)
        resp = requests.put(
            f"http://{rpc_server}/data/{actual_shard_id}",
            data=orjson.dumps(serialized),
        )
        assert resp.status_code == 200

        # Localize (fetches remote tensor)
        localized = RTensor.localize(remotized)

        assert isinstance(localized["logits"], torch.Tensor)
        assert torch.allclose(localized["logits"], output["logits"])
        assert localized["score"] == 0.95

    def test_clear_batch_data(self, rpc_server):
        """Test clearing stored tensor shards."""
        # Store some tensors
        shard_ids = []
        for i in range(3):
            tensor = torch.randn(2, 3).cpu()
            shard_id = str(uuid.uuid4())
            shard_ids.append(shard_id)

            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        # Clear the shards
        resp = requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": shard_ids},
        )
        assert resp.status_code == 200
        result = resp.json()
        assert result["status"] == "ok"
        assert result["cleared_count"] == 3

        # Verify shards are gone
        for shard_id in shard_ids:
            resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
            assert resp.status_code == 404

    def test_batch_shard_retrieval(self, rpc_server):
        """Retrieve multiple shards with one HTTP request."""
        tensors = [torch.randn(2, 3).cpu(), torch.randn(4, 5).cpu()]
        shard_ids = [str(uuid.uuid4()) for _ in tensors]

        for shard_id, tensor in zip(shard_ids, tensors):
            serialized = serialize_value(tensor)
            resp = requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )
            assert resp.status_code == 200

        resp = requests.post(
            f"http://{rpc_server}/data/batch",
            json={"shard_ids": shard_ids},
        )
        assert resp.status_code == 200
        serialized_batch = orjson.loads(resp.content)
        localized = deserialize_value(serialized_batch)
        assert len(localized) == len(tensors)
        for actual, expected in zip(localized, tensors):
            assert torch.allclose(actual, expected)

    def test_batch_shard_retrieval_reports_missing_shards(self, rpc_server):
        """Missing shards return a structured client error instead of a compatibility 404."""
        tensor = torch.randn(2, 3).cpu()
        present_shard_id = str(uuid.uuid4())
        missing_shard_id = str(uuid.uuid4())

        resp = requests.put(
            f"http://{rpc_server}/data/{present_shard_id}",
            data=orjson.dumps(serialize_value(tensor)),
        )
        assert resp.status_code == 200

        resp = requests.post(
            f"http://{rpc_server}/data/batch",
            json={"shard_ids": [present_shard_id, missing_shard_id]},
        )
        assert resp.status_code == 400
        payload = resp.json()
        assert payload["status"] == "error"
        assert payload["missing_shard_ids"] == [missing_shard_id]


class TestHttpRTensorBackendBatching:
    """Unit tests for HTTP batch fetching behavior."""

    def test_fetch_chunks_large_requests(self, monkeypatch):
        """Large same-node fetches are split into bounded batch requests."""
        backend = HttpRTensorBackend(max_shards_per_request=2)
        shards = [
            TensorShardInfo(shard_id=f"s{i}", node_addr="node-a") for i in range(5)
        ]
        requested_chunks = []

        class _FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        async def fake_fetch_shard_group(self, session, node_addr, grouped):
            requested_chunks.append(
                (node_addr, [shard.shard_id for _, shard in grouped])
            )
            return [torch.tensor([int(shard.shard_id[1:])]) for _, shard in grouped]

        monkeypatch.setattr(
            backend,
            "_create_session",
            lambda: _FakeSession(),
        )
        monkeypatch.setattr(
            backend,
            "_fetch_shard_group",
            fake_fetch_shard_group.__get__(backend, HttpRTensorBackend),
        )

        results = backend.fetch(shards)

        assert requested_chunks == [
            ("node-a", ["s0", "s1"]),
            ("node-a", ["s2", "s3"]),
            ("node-a", ["s4"]),
        ]
        assert [int(tensor.item()) for tensor in results] == [0, 1, 2, 3, 4]

    def test_fetch_shard_group_raises_on_missing_batch_endpoint(self):
        """404 on /data/batch surfaces as an error."""
        backend = HttpRTensorBackend()
        grouped = [
            (0, TensorShardInfo(shard_id="s0", node_addr="node-a")),
            (1, TensorShardInfo(shard_id="s1", node_addr="node-a")),
        ]

        class _FakeResponse:
            status = 404

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def text(self):
                return "missing endpoint"

        class _FakeSession:
            def post(self, url, json):
                assert url == "http://node-a/data/batch"
                assert json == {"shard_ids": ["s0", "s1"]}
                return _FakeResponse()

        with pytest.raises(
            RuntimeError,
            match="Failed to fetch shard batch from http://node-a/data/batch: 404 body=missing endpoint",
        ):
            asyncio.run(backend._fetch_shard_group(_FakeSession(), "node-a", grouped))


class TestRTensorErrorHandling:
    """Test error handling for network and storage failures."""

    def test_to_local_with_missing_shard(self, rpc_server):
        """RuntimeError on HTTP 404."""
        rtensor = RTensor(
            shard=TensorShardInfo(
                shard_id="nonexistent-shard-id",
                node_addr=rpc_server,
            ),
            data=torch.empty(3, 20, device="meta"),
        )

        with pytest.raises(RuntimeError, match="Failed to fetch shard"):
            rtensor.to_local()

    def test_to_local_with_server_error(self, rpc_server):
        """RuntimeError on deleted shard."""
        from areal.infra.rpc.rtensor import remove, store

        tensor = torch.randn(2, 5).cpu()
        shard_id = str(uuid.uuid4())
        store(shard_id, tensor)

        rtensor = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(2, 5, device="meta"),
        )

        remove(shard_id)

        with pytest.raises(RuntimeError):
            rtensor.to_local()


class TestRTensorConcurrency:
    """Test concurrent operations on storage."""

    def test_concurrent_storage_writes(self, rpc_server):
        """20 threads store different shards."""
        import threading

        shard_ids = [str(uuid.uuid4()) for _ in range(20)]
        tensors = [torch.randn(2, 3).cpu() for _ in range(20)]

        def store_shard(shard_id, tensor):
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        threads = [
            threading.Thread(target=store_shard, args=(sid, t))
            for sid, t in zip(shard_ids, tensors)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all shards retrievable
        for shard_id in shard_ids:
            resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
            assert resp.status_code == 200

    def test_concurrent_storage_reads(self, rpc_server):
        """10 threads fetch same shard."""
        import threading

        tensor = torch.randn(5, 8).cpu()
        shard_id = str(uuid.uuid4())
        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
        )

        results = [None] * 10

        def fetch_shard(idx):
            rtensor = RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id,
                    node_addr=rpc_server,
                ),
                data=torch.empty(5, 8, device="meta"),
            )
            results[idx] = rtensor.to_local()

        threads = [threading.Thread(target=fetch_shard, args=(i,)) for i in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all fetched tensors identical
        for result in results:
            assert torch.allclose(result, tensor)

    def test_concurrent_clear_operations(self, rpc_server):
        """3 threads clear overlapping shards."""
        import threading

        shard_ids = [str(uuid.uuid4()) for _ in range(10)]
        for shard_id in shard_ids:
            tensor = torch.randn(2, 2).cpu()
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
            )

        # Overlapping shard sets
        shard_sets = [
            shard_ids[:5],
            shard_ids[3:8],
            shard_ids[6:],
        ]

        def clear_shards(shard_list):
            requests.delete(
                f"http://{rpc_server}/data/clear",
                json={"shard_ids": shard_list},
            )

        threads = [threading.Thread(target=clear_shards, args=(s,)) for s in shard_sets]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify all shards deleted (no errors)
        for shard_id in shard_ids:
            resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
            assert resp.status_code == 404


class TestRTensorComplexPadding:
    """Test padding with complex tensor shapes."""

    def test_localize_with_3d_nested_padding(self, rpc_server):
        """Nested structures with 3D tensors."""
        tensor1 = torch.randn(2, 5, 16).cpu()
        tensor2 = torch.randn(3, 8, 16).cpu()

        shard_id1 = str(uuid.uuid4())
        shard_id2 = str(uuid.uuid4())

        for shard_id, tensor in [(shard_id1, tensor1), (shard_id2, tensor2)]:
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}",
                data=orjson.dumps(serialized),
            )

        nested = {
            "encoder": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id1,
                    node_addr=rpc_server,
                ),
                data=torch.empty(2, 5, 16, device="meta"),
            ),
            "decoder": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_id2,
                    node_addr=rpc_server,
                ),
                data=torch.empty(3, 8, 16, device="meta"),
            ),
        }

        localized = RTensor.localize(nested)

        assert isinstance(localized["encoder"], torch.Tensor)
        assert isinstance(localized["decoder"], torch.Tensor)
        assert torch.allclose(localized["encoder"], tensor1)
        assert torch.allclose(localized["decoder"], tensor2)


class TestRTensorEdgeCases:
    """Test edge cases like empty batches and single-item batches."""

    def test_remotize_with_none_values(self):
        """None preserved in structures."""
        obj = {"logits": torch.randn(4, 10).cpu(), "mask": None, "score": 0.95}

        remotized = RTensor.remotize(obj, node_addr="node1")

        assert remotized["mask"] is None
        assert remotized["score"] == 0.95
        assert isinstance(remotized["logits"], RTensor)


class TestRTensorMemoryCleanup:
    """Test memory cleanup and storage stats."""

    def test_storage_stats_accuracy(self, rpc_server):
        """Verify cleared_count and bytes."""
        shard_ids = []
        for i in range(5):
            tensor = torch.randn(10, 20).cpu()
            shard_id = str(uuid.uuid4())
            shard_ids.append(shard_id)
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
            )

        resp = requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": shard_ids},
        )
        result = resp.json()

        assert result["status"] == "ok"
        assert result["cleared_count"] == 5

    def test_clear_batches_nested_structure(self, rpc_server):
        """collect_shards on nested dict."""
        tensors = [torch.randn(3, 5).cpu(), torch.randn(2, 4).cpu()]
        shard_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        for shard_id, tensor in zip(shard_ids, tensors):
            serialized = serialize_value(tensor)
            requests.put(
                f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
            )

        nested = {
            "batch1": RTensor(
                shard=TensorShardInfo(
                    shard_id=shard_ids[0],
                    node_addr=rpc_server,
                ),
                data=torch.empty(3, 5, device="meta"),
            ),
            "batch2": {
                "inner": RTensor(
                    shard=TensorShardInfo(
                        shard_id=shard_ids[1],
                        node_addr=rpc_server,
                    ),
                    data=torch.empty(2, 4, device="meta"),
                )
            },
        }

        shards_by_node = RTensor.collect_shards(nested)
        assert rpc_server in shards_by_node
        assert set(shards_by_node[rpc_server]) == set(shard_ids)

        # Clear all shards
        resp = requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": shard_ids},
        )
        assert resp.status_code == 200

        # Verify deletion
        for shard_id in shard_ids:
            resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
            assert resp.status_code == 404

    def test_storage_cleanup_after_localize(self, rpc_server):
        """Shards persist after fetch."""
        tensor = torch.randn(4, 6).cpu()
        shard_id = str(uuid.uuid4())
        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}", data=orjson.dumps(serialized)
        )

        rtensor = RTensor(
            shard=TensorShardInfo(
                shard_id=shard_id,
                node_addr=rpc_server,
            ),
            data=torch.empty(4, 6, device="meta"),
        )

        localized = rtensor.to_local()
        assert torch.allclose(localized, tensor)

        # Verify shard still on server (not auto-deleted)
        resp = requests.get(f"http://{rpc_server}/data/{shard_id}")
        assert resp.status_code == 200


class TestRemotize:
    """Test remotize method with various input types."""

    def test_remotize_list_of_dicts(self, rpc_server):
        """Test remotizing list of dicts with different attention masks."""
        # Create two trajectory dicts with different seqlens
        traj1 = {
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "input_ids": torch.randn(2, 4),
            "logits": torch.randn(2, 4).cpu(),
        }
        traj2 = {
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
            ),
            "input_ids": torch.randn(3, 5),
            "logits": torch.randn(3, 5).cpu(),
        }

        result = RTensor.remotize([traj1, traj2], node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)
        assert isinstance(result[0]["logits"], RTensor)
        assert isinstance(result[1]["logits"], RTensor)
        # Verify different shard_ids (per-trajectory isolation)
        assert result[0]["logits"].shard.shard_id != result[1]["logits"].shard.shard_id
        # Verify size matches batch dimension
        assert result[0]["logits"].shape[0] == 2
        assert result[1]["logits"].shape[0] == 3

    def test_remotize_list_of_tensors(self, rpc_server):
        """Test remotizing list of standalone tensors."""
        tensors = [torch.randn(2, 5).cpu(), torch.randn(3, 7).cpu()]

        result = RTensor.remotize(tensors, node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, RTensor) for r in result)
        assert result[0].shape[0] == 2
        assert result[0].data.shape == torch.Size([2, 5])
        assert result[1].shape[0] == 3
        assert result[1].data.shape == torch.Size([3, 7])

    def test_remotize_list_with_none(self, rpc_server):
        """Test remotizing list with None values interspersed."""
        traj_dict = {
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "logits": torch.randn(1, 4).cpu(),
        }

        result = RTensor.remotize([traj_dict, None, traj_dict], node_addr=rpc_server)

        assert isinstance(result, list)
        assert len(result) == 3
        assert isinstance(result[0], dict)
        assert result[1] is None
        assert isinstance(result[2], dict)
        assert isinstance(result[0]["logits"], RTensor)
        assert isinstance(result[2]["logits"], RTensor)

    def test_remotize_single_dict(self, rpc_server):
        """Test remotizing single dict (not wrapped in list)."""
        traj_dict = {
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
            "logits": torch.randn(1, 4).cpu(),
        }

        result = RTensor.remotize(traj_dict, node_addr=rpc_server)

        assert isinstance(result, dict)
        assert isinstance(result["logits"], RTensor)
        assert result["logits"].shape[0] == 1

    def test_remotize_standalone_tensor(self, rpc_server):
        """Test remotizing standalone tensor (not in dict or list)."""
        tensor = torch.randn(2, 5).cpu()

        result = RTensor.remotize(tensor, node_addr=rpc_server)

        assert isinstance(result, RTensor)
        assert result.shape[0] == 2
        assert result.data.shape == torch.Size([2, 5])

    def test_remotize_none(self):
        """Test that None input returns None."""
        result = RTensor.remotize(None, node_addr="localhost:8080")
        assert result is None

    def test_remotize_scalar(self):
        """Test that scalar values pass through unchanged."""
        result_int = RTensor.remotize(42, node_addr="localhost:8080")
        assert result_int == 42

        result_bool = RTensor.remotize(True, node_addr="localhost:8080")
        assert result_bool is True

        result_float = RTensor.remotize(3.14, node_addr="localhost:8080")
        assert result_float == 3.14

    def test_remotize_float_dict(self):
        """Test that dict without tensors returns with values unchanged."""
        obj = {"lr": 0.001, "grad_norm": 1.5}
        result = RTensor.remotize(obj, node_addr="localhost:8080")
        assert result["lr"] == 0.001
        assert result["grad_norm"] == 1.5

    def test_remotize_empty_list(self):
        """Test that empty list returns empty list."""
        result = RTensor.remotize([], node_addr="localhost:8080")
        assert result == []

    def test_remotize_roundtrip(self, rpc_server):
        """Test remotize->localize roundtrip for trajectory dict."""
        original_traj = {
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "logits": torch.randn(2, 4).cpu(),
        }
        original_logits = original_traj["logits"].clone()

        # Remotize
        remotized = RTensor.remotize(original_traj, node_addr=rpc_server)

        # Store tensors on server using the NEW shard_ids created by remotize
        from areal.infra.rpc.rtensor import fetch

        for key in ["attention_mask", "logits"]:
            if isinstance(remotized[key], RTensor):
                actual_shard_id = remotized[key].shard.shard_id
                tensor_from_local = fetch(actual_shard_id)
                serialized = serialize_value(tensor_from_local)
                resp = requests.put(
                    f"http://{rpc_server}/data/{actual_shard_id}",
                    data=orjson.dumps(serialized),
                )
                assert resp.status_code == 200

        # Localize (fetches remote tensors)
        localized = RTensor.localize(remotized)

        assert isinstance(localized["logits"], torch.Tensor)
        # After unpadding, logits are trimmed to max seqlen=3 (from attention_mask)
        assert localized["logits"].shape == (2, 3)
        assert torch.allclose(localized["logits"], original_logits[:, :3], atol=1e-5)

    def test_remotize_trims_padding_from_attention_mask(self, rpc_server):
        """Verify remotize trims padding when dict has attention_mask.

        Create a dict with attention_mask [[1,1,1,0,0], [1,1,0,0,0]] (seqlen 5,
        actual max 3). Remotize and localize. Assert tensors trimmed to seqlen 3.
        """
        traj = {
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]),
            "input_ids": torch.randn(2, 5),
            "logits": torch.randn(2, 5),
        }

        remotized = RTensor.remotize(traj, node_addr=rpc_server)

        # All tensor values should be RTensors
        assert isinstance(remotized["attention_mask"], RTensor)
        assert isinstance(remotized["input_ids"], RTensor)
        assert isinstance(remotized["logits"], RTensor)

        # The data (meta tensor) should reflect the compacted shape
        assert remotized["attention_mask"].data.shape == torch.Size([2, 3])
        assert remotized["input_ids"].data.shape == torch.Size([2, 3])
        assert remotized["logits"].data.shape == torch.Size([2, 3])

        # Verify via localize roundtrip
        from areal.infra.rpc.rtensor import fetch

        for key in ["attention_mask", "input_ids", "logits"]:
            actual_shard_id = remotized[key].shard.shard_id
            tensor_from_local = fetch(actual_shard_id)
            serialized = serialize_value(tensor_from_local)
            resp = requests.put(
                f"http://{rpc_server}/data/{actual_shard_id}",
                data=orjson.dumps(serialized),
            )
            assert resp.status_code == 200

        localized = RTensor.localize(remotized)
        assert localized["attention_mask"].shape == (2, 3)
        assert localized["input_ids"].shape == (2, 3)
        assert localized["logits"].shape == (2, 3)
        # attention_mask should be trimmed to [[1,1,1],[1,1,0]]
        expected_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        assert torch.equal(localized["attention_mask"], expected_mask)


class TestFetchBuffer:
    """Test client-side fetch buffer for RTensor caching.

    The fetch buffer avoids redundant network fetches when the same
    rollout_batch is sent to multiple engine calls across RPC boundaries.
    """

    def setup_method(self):
        """Clear fetch buffer before each test."""
        from areal.infra.rpc.rtensor import _fetch_buffer, _fetch_buffer_lock

        with _fetch_buffer_lock:
            _fetch_buffer.clear()

    def test_to_local_populates_buffer(self, rpc_server):
        """to_local() should populate the fetch buffer on first access."""
        from areal.infra.rpc.rtensor import _fetch_buffer

        tensor = torch.randn(3, 5).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        rtensor = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(3, 5, device="meta"),
        )

        result = rtensor.to_local()
        assert torch.allclose(result, tensor)
        assert shard_id in _fetch_buffer

    def test_to_local_serves_from_buffer(self, rpc_server):
        """Second to_local() with a fresh RTensor (same shard_id) should
        hit the buffer without making a network request."""
        tensor = torch.randn(4, 6).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        # First access: populates buffer
        rt1 = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(4, 6, device="meta"),
        )
        result1 = rt1.to_local()

        # Delete shard from server so a real fetch would fail
        requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": [shard_id]},
        )

        # Second access with a new RTensor object (simulates RPC boundary)
        rt2 = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(4, 6, device="meta"),
        )
        result2 = rt2.to_local()
        assert torch.allclose(result1, result2)

    def test_localize_populates_buffer(self, rpc_server):
        """localize() should populate the fetch buffer for all fetched shards."""
        from areal.infra.rpc.rtensor import _fetch_buffer

        tensor1 = torch.randn(2, 3).cpu()
        tensor2 = torch.randn(4, 5).cpu()
        shard_id1 = str(uuid.uuid4())
        shard_id2 = str(uuid.uuid4())

        for sid, t in [(shard_id1, tensor1), (shard_id2, tensor2)]:
            serialized = serialize_value(t)
            requests.put(
                f"http://{rpc_server}/data/{sid}",
                data=orjson.dumps(serialized),
            )

        nested = {
            "a": RTensor(
                shard=TensorShardInfo(shard_id=shard_id1, node_addr=rpc_server),
                data=torch.empty(2, 3, device="meta"),
            ),
            "b": RTensor(
                shard=TensorShardInfo(shard_id=shard_id2, node_addr=rpc_server),
                data=torch.empty(4, 5, device="meta"),
            ),
        }

        localized = RTensor.localize(nested)
        assert torch.allclose(localized["a"], tensor1)
        assert torch.allclose(localized["b"], tensor2)
        assert shard_id1 in _fetch_buffer
        assert shard_id2 in _fetch_buffer

    def test_localize_serves_from_buffer(self, rpc_server):
        """Second localize() with fresh meta RTensors (same shard_ids) should
        resolve entirely from the buffer."""
        tensor = torch.randn(3, 4).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        def _make_rtensor():
            return RTensor(
                shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
                data=torch.empty(3, 4, device="meta"),
            )

        # First localize: populates buffer
        result1 = RTensor.localize({"x": _make_rtensor()})

        # Remove from server
        requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": [shard_id]},
        )

        # Second localize with fresh meta RTensor: should hit buffer
        result2 = RTensor.localize({"x": _make_rtensor()})
        assert torch.allclose(result1["x"], result2["x"])

    def test_localize_partial_buffer_hit(self, rpc_server):
        """When some shards are in the buffer and others are not, only the
        misses should be fetched from the backend."""
        from areal.infra.rpc.rtensor import _fetch_buffer

        tensor_a = torch.randn(2, 3).cpu()
        tensor_b = torch.randn(4, 5).cpu()
        shard_a = str(uuid.uuid4())
        shard_b = str(uuid.uuid4())

        for sid, t in [(shard_a, tensor_a), (shard_b, tensor_b)]:
            serialized = serialize_value(t)
            requests.put(
                f"http://{rpc_server}/data/{sid}",
                data=orjson.dumps(serialized),
            )

        # Warm buffer with shard_a only
        rt_a = RTensor(
            shard=TensorShardInfo(shard_id=shard_a, node_addr=rpc_server),
            data=torch.empty(2, 3, device="meta"),
        )
        RTensor.localize(rt_a)
        assert shard_a in _fetch_buffer
        assert shard_b not in _fetch_buffer

        # Delete shard_a from server; shard_b remains
        requests.delete(
            f"http://{rpc_server}/data/clear",
            json={"shard_ids": [shard_a]},
        )

        # Localize both: shard_a from buffer, shard_b from backend
        nested = {
            "a": RTensor(
                shard=TensorShardInfo(shard_id=shard_a, node_addr=rpc_server),
                data=torch.empty(2, 3, device="meta"),
            ),
            "b": RTensor(
                shard=TensorShardInfo(shard_id=shard_b, node_addr=rpc_server),
                data=torch.empty(4, 5, device="meta"),
            ),
        }
        result = RTensor.localize(nested)
        assert torch.allclose(result["a"], tensor_a)
        assert torch.allclose(result["b"], tensor_b)

    def test_clear_node_evicts_from_buffer(self, rpc_server):
        """clear_node() should remove entries from the fetch buffer."""
        from areal.infra.rpc.rtensor import _fetch_buffer

        tensor = torch.randn(2, 3).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        # Populate buffer
        rt = RTensor(
            shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
            data=torch.empty(2, 3, device="meta"),
        )
        rt.to_local()
        assert shard_id in _fetch_buffer

        # clear_node evicts from buffer
        asyncio.run(RTensor.clear_node(rpc_server, [shard_id]))
        assert shard_id not in _fetch_buffer

    def test_clear_fetch_buffer_selective(self):
        """clear_fetch_buffer(sids) pops only the listed entries, misses are no-ops."""
        from areal.infra.rpc.rtensor import _fetch_buffer, clear_fetch_buffer

        _fetch_buffer["a"] = torch.tensor([1])
        _fetch_buffer["b"] = torch.tensor([2])
        _fetch_buffer["c"] = torch.tensor([3])

        removed = clear_fetch_buffer(["a", "missing", "c"])
        assert removed == 2
        assert "a" not in _fetch_buffer
        assert "b" in _fetch_buffer
        assert "c" not in _fetch_buffer

    def test_clear_fetch_buffer_flush_all(self):
        """clear_fetch_buffer(None) drops every entry."""
        from areal.infra.rpc.rtensor import _fetch_buffer, clear_fetch_buffer

        _fetch_buffer["a"] = torch.tensor([1])
        _fetch_buffer["b"] = torch.tensor([2])

        removed = clear_fetch_buffer(None)
        assert removed == 2
        assert len(_fetch_buffer) == 0

    def test_fetch_buffer_stats_reports_entry_count(self):
        """fetch_buffer_stats() returns {'num_entries': N} matching buffer size."""
        from areal.infra.rpc.rtensor import _fetch_buffer, fetch_buffer_stats

        assert fetch_buffer_stats() == {"num_entries": 0}
        _fetch_buffer["a"] = torch.tensor([1])
        _fetch_buffer["b"] = torch.tensor([2])
        assert fetch_buffer_stats() == {"num_entries": 2}

    def test_remove_pops_both_storage_and_fetch_buffer(self):
        """remove() drops the shard from both _storage and _fetch_buffer.

        Regression guard for areal-project/AReaL#1209: on a worker that is both
        storage owner and consumer, ``to_local()`` caches the tensor in
        ``_fetch_buffer``; without this pop, RSS grows unboundedly across
        training steps.
        """
        from areal.infra.rpc.rtensor import (
            _fetch_buffer,
            _storage,
            _store_local,
            remove,
        )

        sid = "test-sid-1209-regression"
        tensor = torch.arange(6).reshape(2, 3)
        _store_local(sid, tensor)
        _fetch_buffer[sid] = tensor  # simulate to_local() having cached it

        assert sid in _storage
        assert sid in _fetch_buffer

        n = remove(sid)
        assert n == 1
        assert sid not in _storage
        assert sid not in _fetch_buffer

    def test_buffer_thread_safety(self, rpc_server):
        """Concurrent to_local() calls with the same shard_id should not crash."""
        import threading

        tensor = torch.randn(5, 8).cpu()
        shard_id = str(uuid.uuid4())

        serialized = serialize_value(tensor)
        requests.put(
            f"http://{rpc_server}/data/{shard_id}",
            data=orjson.dumps(serialized),
        )

        results = [None] * 10

        def fetch_shard(idx):
            rt = RTensor(
                shard=TensorShardInfo(shard_id=shard_id, node_addr=rpc_server),
                data=torch.empty(5, 8, device="meta"),
            )
            results[idx] = rt.to_local()

        threads = [threading.Thread(target=fetch_shard, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for result in results:
            assert result is not None
            assert torch.allclose(result, tensor)


class TestFlattenShardIds:
    """Tests for flatten_shard_ids helper used by TrainController.clear_batches.

    The helper walks a nested (dict / list / tuple) structure and returns a flat
    list of every RTensor's shard_id. It is the payload builder for the
    replicated ``clear_batches`` RPC fan-out — sending a flat ``list[str]``
    rather than the original RTensor structure is required so that
    ``engine_blueprint``'s ``RTensor.localize`` pass on the receiving end does
    not re-fetch the shards we are trying to clear.
    """

    @staticmethod
    def _make_rt(sid: str) -> RTensor:
        return RTensor(
            shard=TensorShardInfo(shard_id=sid, node_addr="127.0.0.1:0"),
            data=torch.empty(1, device="meta"),
        )

    def test_empty_structures_return_empty_list(self):
        from areal.infra.rpc.rtensor import flatten_shard_ids

        assert flatten_shard_ids(()) == []
        assert flatten_shard_ids({}) == []
        assert flatten_shard_ids([]) == []
        assert flatten_shard_ids(None) == []

    def test_structure_without_rtensor_returns_empty_list(self):
        from areal.infra.rpc.rtensor import flatten_shard_ids

        assert flatten_shard_ids({"x": 1, "y": "str"}) == []
        assert flatten_shard_ids([torch.tensor([1, 2])]) == []

    def test_nested_dict_list_tuple_are_all_walked(self):
        from areal.infra.rpc.rtensor import flatten_shard_ids

        rt_a = self._make_rt("sid-a")
        rt_b = self._make_rt("sid-b")
        rt_c = self._make_rt("sid-c")

        obj = (
            {"x": rt_a, "nested": [rt_b, {"y": rt_c}]},
            {"batch_id": rt_a},  # intentional repeat across top-level args
        )
        sids = flatten_shard_ids(obj)
        assert set(sids) == {"sid-a", "sid-b", "sid-c"}
        # collect_shards preserves duplicates; downstream pop handles misses.
        assert sids.count("sid-a") == 2


class TestTensorShardInfoDocumentation:
    """Tests verifying TensorShardInfo construction and field semantics."""

    def test_construction_with_all_fields(self):
        """TensorShardInfo can be constructed with required fields."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            shard_id="test-shard-001",
            node_addr="localhost:8080",
        )
        assert shard.shard_id == "test-shard-001"
        assert shard.node_addr == "localhost:8080"

    def test_ray_backend_empty_node_addr(self):
        """Ray backend uses empty string for node_addr."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            shard_id="",  # Will be filled by Ray ObjectRef
            node_addr="",  # Empty for Ray backend
        )
        assert shard.node_addr == ""

    def test_http_backend_node_addr(self):
        """HTTP backend uses host:port for node_addr."""
        from areal.infra.rpc.rtensor import TensorShardInfo

        shard = TensorShardInfo(
            shard_id="some-uuid",
            node_addr="192.168.1.1:8080",
        )
        assert ":" in shard.node_addr
