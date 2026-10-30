# SPDX-License-Identifier: Apache-2.0

"""Normal-path tests for event-driven decisions and the runnable example."""

import json
from types import SimpleNamespace

import httpx
import pytest

from examples.elastic.agent import ControlLLM

from areal.elastic.policy import LifecyclePolicy


def test_agent_wait_coalesces_polling_and_wakes_on_changed_event(tmp_path, monkeypatch):
    """Hundreds of coordinator polls must not become hundreds of LLM calls."""
    clock = [10.0]
    monkeypatch.setattr("areal.elastic.policy.time.monotonic", lambda: clock[0])
    observations = []

    def decide(observation):
        observations.append(observation)
        return {
            "action": "wait_for_event",
            "wake_after_seconds": 30,
            "reason": "Wait for another event",
        }

    policy = LifecyclePolicy(mode="agent", decide=decide)
    owner = SimpleNamespace(
        root=tmp_path,
        actor=None,
        options=SimpleNamespace(idle_timeout=30, min_ready_seconds=0),
    )
    status = {
        "generation": 0,
        "applied_version": 0,
        "counts": {"ready": 4},
        "idle_seconds": 0,
        "residence_seconds": 0,
    }
    owner.get_status = lambda **kwargs: dict(status)
    policy.bind(owner)
    batch = SimpleNamespace(batch_id="first")
    for _ in range(200):
        assert policy.decide(batch) == "keep"
    assert len(observations) == 1
    status["counts"] = {"ready": 8}
    policy.decide(batch)
    assert len(observations) == 2
    clock[0] = 41
    policy.decide(batch)
    assert len(observations) == 3
    assert "future_arrivals" not in observations[0]


def test_agent_training_does_not_call_llm_for_each_batch(tmp_path):
    calls = []
    policy = LifecyclePolicy(mode="agent", decide=calls.append)
    owner = SimpleNamespace(root=tmp_path, actor=object())
    policy.bind(owner)
    for number in range(20):
        assert policy.decide(SimpleNamespace(batch_id=str(number))) == "train"
    assert calls == []


def test_control_llm_reads_status_then_requests_start(monkeypatch):
    """Exercise the actual HTTP request/tool adapter with a deterministic endpoint."""
    monkeypatch.setenv("AREAL_CONTROL_LLM_BASE_URL", "https://controller.invalid/v1")
    monkeypatch.setenv("AREAL_CONTROL_LLM_MODEL", "test-controller")
    monkeypatch.setenv("AREAL_CONTROL_LLM_API_KEY", "test-only-key")
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        action = "get_status" if len(requests) == 1 else "start_actor"
        args = {} if action == "get_status" else {"reason": "A complete batch is ready"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": str(len(requests)),
                                    "type": "function",
                                    "function": {
                                        "name": action,
                                        "arguments": json.dumps(args),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        llm = ControlLLM(client=http)
        decision = llm.decide({"event": "batch_ready", "generation": 0})
    assert decision["action"] == "start_actor"
    assert len(requests) == 2
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert "test-only-key" not in json.dumps(requests)


@pytest.mark.parametrize("replicas", [2, 4])
def test_off_progress_uses_lifecycle_boundaries_when_poll_misses_batch(
    tmp_path, replicas
):
    from examples.elastic.run_demo import summarize

    (tmp_path / "state").mkdir()
    events = [
        {"state": "TRAINING", "generation": 1, "applied_version": 1},
        {
            "state": "STOPPED",
            "generation": 1,
            "applied_version": 2,
            "monotonic": 10,
            "actor_gpu_seconds": 100,
            "rollout": {"accepted": 8},
        },
        {
            "state": "STARTING",
            "generation": 2,
            "applied_version": 2,
            "monotonic": 20,
            "rollout": {"accepted": 12},
        },
        {"state": "RESTORED", "generation": 2, "applied_version": 2},
        {"state": "TRAINING", "generation": 2, "applied_version": 2},
    ]
    (tmp_path / "state/events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    manager = SimpleNamespace(
        policy_adapter=SimpleNamespace(mode="agent"),
        applied_version=3,
        rollout=SimpleNamespace(serving_versions=lambda: [3] * replicas),
        data_progress={"samples": 12},
    )
    result = summarize(tmp_path, manager, 3)
    assert (
        result["rollout_progress_while_actor_off"] and result["elastic_cycle_observed"]
    )


def test_retire_checkpoint_requires_closed_run_and_preserves_other_artifacts(tmp_path):
    import tarfile

    from examples.elastic.retire_checkpoint import retire_checkpoint

    checkpoint = tmp_path / "state/checkpoints"
    (checkpoint / "step/dcp").mkdir(parents=True)
    (checkpoint / "latest.json").write_text("{}")
    (checkpoint / "step/manifest.json").write_text("{}")
    (checkpoint / "step/dcp/weights.distcp").write_bytes(b"checkpoint tensors")
    (tmp_path / "result.json").write_text(json.dumps({"status": "training_completed"}))
    (tmp_path / "model.safetensors").write_bytes(b"keep the source model")
    status_path = tmp_path / "state/status.json"
    status_path.write_text(
        json.dumps({"coordinator_closed": False, "actor_running": False})
    )
    with pytest.raises(ValueError, match="closed coordinator"):
        retire_checkpoint(tmp_path)
    assert checkpoint.exists()
    status_path.write_text(
        json.dumps({"coordinator_closed": True, "actor_running": False})
    )
    result = retire_checkpoint(tmp_path)
    assert not result["full_resume_available"] and not checkpoint.exists()
    assert (tmp_path / "model.safetensors").read_bytes() == b"keep the source model"
    with tarfile.open(tmp_path / result["metadata_archive"]) as stream:
        assert "step/manifest.json" in stream.getnames()
        assert "step/dcp/weights.distcp" not in stream.getnames()
    assert retire_checkpoint(tmp_path) == result


def test_gpu_samples_preserve_units_and_report_unavailable_metrics(
    tmp_path, monkeypatch
):
    import subprocess

    from examples.elastic.telemetry import GPUMonitor

    def query(command, **kwargs):
        assert "--id=4,5" in command
        return SimpleNamespace(
            stdout="4, GPU-a, 1024, 97280, 40, 12\n5, GPU-b, 2048, 97280, N/A, [N/A]\n"
        )

    monkeypatch.setattr("examples.elastic.telemetry.subprocess.run", query)
    monitor = GPUMonitor(tmp_path / "gpus.jsonl", [4, 5])
    sample = monitor.sample()
    assert sample["gpus"][0]["memory_used_mib"] == 1024
    assert sample["gpus"][0]["gpu_utilization_percent"] == 40
    assert sample["gpus"][1]["gpu_utilization_percent"] is None

    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr("examples.elastic.telemetry.subprocess.run", unavailable)
    sample = monitor.sample()
    assert "TimeoutExpired" in sample["error"] and "gpus" not in sample
