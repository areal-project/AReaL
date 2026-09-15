# SPDX-License-Identifier: Apache-2.0

"""Exercise evaluation ownership with real RTensor storage and local RPC transport."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from flask.testing import FlaskClient

from areal.infra.controller.train_controller import TrainController
from areal.infra.rpc import rtensor as rt
from areal.infra.rpc.guard import engine_blueprint as eb
from areal.infra.rpc.guard.app import GuardState, create_app
from areal.infra.rpc.guard.data_blueprint import data_bp
from areal.infra.rpc.serialization import deserialize_value
from areal.trainer import dpo_trainer, rl_trainer
from areal.trainer.dpo_trainer import DPOTrainer
from areal.trainer.rl_trainer import PPOTrainer


def _trajectory() -> dict[str, Any]:
    return {
        "input_ids": torch.ones((1, 4), dtype=torch.int64, device="cpu"),
        "attention_mask": torch.ones((1, 4), dtype=torch.bool, device="cpu"),
        "rewards": torch.ones((1,), dtype=torch.float32, device="cpu"),
        "multi_modal_input": {
            "pixel_values": torch.ones((1, 3, 2, 2), device="cpu", dtype=torch.float32),
        },
    }


class _CompletedRollout:
    def wait_for_task(self, task_id: int) -> dict[str, Any] | None:
        # Include filtered results alongside nested multimodal trajectories.
        return _trajectory() if task_id % 2 == 0 else None


class _EvalRollout:
    def __init__(self, client: FlaskClient) -> None:
        self.client = client
        self.pending: list[int] = []
        self.next_id = 0

    def submit(self, item: Any, workflow: Any, workflow_kwargs: Any, **kwargs) -> None:
        assert kwargs["is_eval"] is True
        self.pending.append(self.next_id)
        self.next_id += 1

    def wait(self, count: int, timeout: float | None = None) -> list[Any]:
        assert count == len(self.pending) and count > 0
        task_ids, self.pending = self.pending, []
        results = []
        for task_id in task_ids:
            response = self.client.post(
                "/call",
                json={
                    "engine_name": "eval-fixture",
                    "method": "wait_for_task",
                    "args": [task_id],
                    "rpc_meta": {"broadcast": False},
                },
            )
            assert response.status_code == 200, response.get_data(as_text=True)
            result = deserialize_value(response.get_json()["result"])
            if result is not None:
                assert isinstance(
                    result["multi_modal_input"]["pixel_values"], rt.RTensor
                )
            results.append(result)
        return results


@pytest.fixture
def rpc_storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[FlaskClient]:
    """Keep storage and serialization real, replacing only HTTP transport."""
    monkeypatch.setenv("AREAL_SPMD_MODE", "0")
    monkeypatch.setattr(rt, "_storage", {})
    monkeypatch.setattr(rt, "_storage_stats", {})
    monkeypatch.setattr(rt, "_fetch_buffer", {})
    monkeypatch.setattr(eb, "_engines", {"eval-fixture": _CompletedRollout()})
    app = create_app(GuardState())
    app.register_blueprint(eb.engine_bp)
    app.register_blueprint(data_bp)
    client = app.test_client()

    async def delete(node_addr: str, shard_ids: list[str]) -> dict[str, Any]:
        response = client.delete("/data/clear", json={"shard_ids": shard_ids})
        assert response.status_code == 200
        return response.get_json()

    backend = rt.HttpRTensorBackend()
    monkeypatch.setattr(backend, "delete", delete)
    monkeypatch.setattr(
        backend, "fetch", lambda shards: [rt.fetch(s.shard_id) for s in shards]
    )
    monkeypatch.setattr(rt, "_backend", backend)
    try:
        yield client
    finally:
        eb.cleanup_engine_thread()


def _controller(monkeypatch: pytest.MonkeyPatch) -> TrainController:
    controller = TrainController(
        train_engine=Mock(),
        config=SimpleNamespace(backend="fsdp:d1"),
        scheduler=Mock(),
    )

    def worker_call(method: str, *args: Any, **kwargs: Any) -> dict[str, int] | None:
        assert kwargs["rpc_meta"] == {"broadcast": False}
        if method == "clear_batches":
            rt.clear_fetch_buffer(args[0])
            return None
        assert method == "fetch_buffer_stats"
        return rt.fetch_buffer_stats()

    monkeypatch.setattr(controller, "_custom_function_call", worker_call)
    monkeypatch.setattr(controller, "is_data_parallel_head", lambda: True)
    return controller


def test_ppo_evaluation_repeated_rounds_release_only_owned_shards(
    rpc_storage: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real guard results are cleared, including nested images, without touching training."""
    training = rt.RTensor.remotize(_trajectory(), node_addr="training")
    training_ids = rt.flatten_shard_ids(training)
    baseline = rt.storage_stats()
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.actor = _controller(monkeypatch)
    trainer.config = SimpleNamespace(eval_gconfig=SimpleNamespace(n_samples=1))
    trainer.valid_dataloader = [[{}] * 4, [{}] * 2]
    trainer.eval_rollout = _EvalRollout(rpc_storage)

    for _ in range(3):
        trainer._evaluate_fn(eval_workflow=None, eval_workflow_kwargs={})
        assert rt.storage_stats() == baseline
        assert set(rt._storage) == set(training_ids)


def test_ppo_evaluation_empty_loader_does_not_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty validation set cannot submit a zero-count dispatcher wait."""
    monkeypatch.setenv("AREAL_SPMD_MODE", "0")
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.actor = Mock()
    trainer.actor.is_data_parallel_head.return_value = True
    trainer.valid_dataloader = []
    trainer.eval_rollout = Mock()

    trainer._evaluate_fn(eval_workflow=None, eval_workflow_kwargs={})

    trainer.eval_rollout.wait.assert_not_called()
    trainer.actor.clear_batches.assert_not_called()


def test_ppo_evaluation_spmd_uses_local_results_without_remote_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPMD keeps its local tensor and collective behavior."""
    monkeypatch.setenv("AREAL_SPMD_MODE", "1")
    barrier = Mock()
    monkeypatch.setattr(rl_trainer.dist, "barrier", barrier)
    monkeypatch.setattr(rl_trainer.current_platform, "synchronize", Mock())
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.actor = Mock()
    trainer.actor.is_data_parallel_head.return_value = True
    trainer.config = SimpleNamespace(eval_gconfig=SimpleNamespace(n_samples=1))
    trainer.valid_dataloader = [[{}]]
    trainer.eval_rollout = Mock()
    trainer.eval_rollout.wait.return_value = [_trajectory()]

    trainer._evaluate_fn(eval_workflow=None, eval_workflow_kwargs={})

    trainer.actor.clear_batches.assert_not_called()
    barrier.assert_called_once_with(group=trainer.actor.cpu_group)


def _dpo_trainer(monkeypatch: pytest.MonkeyPatch) -> DPOTrainer:
    trainer = DPOTrainer.__new__(DPOTrainer)
    trainer.actor = _controller(monkeypatch)
    trainer.ref = _controller(monkeypatch)
    monkeypatch.setattr(dpo_trainer.dist, "barrier", Mock())
    monkeypatch.setattr(dpo_trainer.current_platform, "synchronize", Mock())
    return trainer


@pytest.mark.parametrize("failure", [None, "reference", "actor", "attach"])
def test_dpo_evaluation_releases_batch_and_reference_results_on_failure(
    rpc_storage: FlaskClient, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """Owned source shards and fetch buffers are released, even on partial consumption."""
    training = rt.RTensor.remotize(_trajectory(), node_addr="training")
    training_ids = set(rt.flatten_shard_ids(training))
    trainer = _dpo_trainer(monkeypatch)
    baseline = rt.storage_stats()
    evaluation_error = RuntimeError("evaluation failed")

    def load_batch(_generator: Any) -> list[dict[str, Any]]:
        # Check cleanup happened before requesting another validation batch.
        assert rt.storage_stats() == baseline
        return [rt.RTensor.remotize(_trajectory(), node_addr="validation")]

    def compute_logp(data: Any) -> list[Any]:
        rt.RTensor.localize(data)
        if failure == "reference":
            raise evaluation_error
        values = rt.RTensor.remotize(
            [torch.zeros((1, 4), dtype=torch.float32, device="cpu")],
            node_addr="reference",
        )
        if failure == "attach":
            # Fail before attaching the remaining owned RTensors to data.
            return [None, *values]
        return values

    def evaluate_dpo(data: Any) -> None:
        rt.RTensor.localize(data)
        if failure == "actor":
            raise evaluation_error

    trainer.valid_dataloader = [[{}], [{}]]
    monkeypatch.setattr(trainer, "_load_bcast_from", load_batch)
    monkeypatch.setattr(trainer.ref, "compute_logp", compute_logp, raising=False)
    monkeypatch.setattr(trainer.actor, "evaluate_dpo", evaluate_dpo, raising=False)

    for _ in range(3):
        if failure == "attach":
            with pytest.raises(AttributeError, match="ndim"):
                trainer._evaluate_fn()
        elif failure is not None:
            with pytest.raises(RuntimeError, match="evaluation failed") as exc_info:
                trainer._evaluate_fn()
            assert exc_info.value is evaluation_error
        else:
            trainer._evaluate_fn()
        assert rt.storage_stats() == baseline
        assert set(rt._storage) == training_ids
        assert rt.fetch_buffer_stats() == {"num_entries": 0}


def test_dpo_evaluation_actor_cleanup_failure_still_drains_reference(
    rpc_storage: FlaskClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every consumer is offered cleanup before the first cleanup error propagates."""
    trainer = _dpo_trainer(monkeypatch)
    trainer.valid_dataloader = [[{}]]
    trainer.actor.clear_batches = Mock(side_effect=RuntimeError("actor cleanup failed"))
    reference_clear = Mock(wraps=trainer.ref.clear_batches)
    trainer.ref.clear_batches = reference_clear
    trainer.ref.compute_logp = lambda data: rt.RTensor.remotize(
        [torch.zeros((1, 4), dtype=torch.float32, device="cpu")], node_addr="reference"
    )
    trainer.actor.evaluate_dpo = lambda data: rt.RTensor.localize(data)

    with pytest.raises(RuntimeError, match="actor cleanup failed"):
        trainer._evaluate_fn()

    reference_clear.assert_called_once()
    assert rt.storage_stats() == {"num_tensors": 0, "total_bytes": 0}
    assert rt.fetch_buffer_stats() == {"num_entries": 0}


def test_dpo_evaluation_spmd_uses_local_results_without_remote_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SPMD still attaches local log probabilities and synchronizes after evaluation."""
    monkeypatch.setenv("AREAL_SPMD_MODE", "1")
    barrier = Mock()
    monkeypatch.setattr(dpo_trainer.dist, "barrier", barrier)
    monkeypatch.setattr(dpo_trainer.current_platform, "synchronize", Mock())
    trainer = DPOTrainer.__new__(DPOTrainer)
    trainer.actor = Mock()
    trainer.ref = Mock()
    trainer.valid_dataloader = [[{}]]
    data = [_trajectory()]
    trainer._load_bcast_from = Mock(return_value=data)
    logp = torch.zeros((4,), dtype=torch.float32, device="cpu")
    trainer.ref.compute_logp.return_value = [logp]

    trainer._evaluate_fn()

    torch.testing.assert_close(
        data[0]["ref_logprobs"], logp.unsqueeze(0), rtol=0, atol=0
    )
    trainer.actor.evaluate_dpo.assert_called_once_with(data)
    trainer.actor.clear_batches.assert_not_called()
    trainer.ref.clear_batches.assert_not_called()
    barrier.assert_called_once_with(group=trainer.actor.cpu_group)
