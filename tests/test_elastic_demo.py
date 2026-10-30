# SPDX-License-Identifier: Apache-2.0

"""Normal-path tests for event-driven decisions and the runnable example."""

import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from examples.elastic.agent import ControlLLM

from areal.elastic.policy import LifecyclePolicy


@pytest.mark.parametrize("custom_timing", [False, True])
def test_resident_protocol_preparation_records_lifecycle_estimates(
    tmp_path, monkeypatch, custom_timing
):
    """GPU protocol callers can prepare without agent-only CLI options."""
    from examples.elastic.run_demo import prepare

    executable = tmp_path / "kubectl"
    executable.touch()
    kubeconfig = tmp_path / "kubeconfig"
    kubeconfig.touch()
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "python": sys.executable,
                "pythonpath": str(tmp_path),
                "system_url": "http://sandbox.invalid",
                "environment": "test-swe@0.1.0",
                "kubectl": str(executable),
                "kubeconfig": str(kubeconfig),
                "sandbox_namespace": "test-swe",
            }
        )
    )
    dataset = tmp_path / "tasks.jsonl"
    row = {
        "instance_id": "repo__project-1",
        "problem_statement": "Fix this issue",
        "eval_script": "run-reference-tests",
        "base_commit": "a" * 40,
    }
    dataset.write_text((json.dumps(row) + "\n") * 2)
    args = SimpleNamespace(
        model=tmp_path / "model",
        dataset=dataset,
        swe_runtime=runtime,
        output=tmp_path / "run",
        steps=1,
        policy="resident",
        devices=list(range(8)),
        concurrency=2,
        export_style="concat",
        idle_seconds=40,
        max_turns=24,
        max_tokens=4096,
        context_tokens=16384,
        checkpoint_gib=1,
        prepare_only=True,
    )
    expected = dict(
        startup_estimate_seconds=90,
        checkpoint_estimate_seconds=40,
        release_estimate_seconds=15,
        restore_margin_seconds=15,
    )
    if custom_timing:
        expected = {name: value + 1 for name, value in expected.items()}
        args.__dict__.update(expected)
    for name in (
        "AREAL_DEMO_ROOT",
        "AREAL_MODEL_PATH",
        "AREAL_DATASET_PATH",
        "AREAL_DEMO_POLICY",
        "AREAL_INFERENCE_ADMIN_KEY",
        "AREAL_ACTOR_ADMIN_KEY",
    ):
        monkeypatch.setenv(name, "test-only")

    config, output, tasks = prepare(args)

    timing = json.loads((output / "plan.json").read_text())["agent_timing"]
    assert {name: timing[name] for name in expected} == expected
    assert config.actor.backend == "megatron:d1t4"
    assert config.rollout.backend == "sglang:d2t2"
    assert len(tasks) == config.train_dataset.batch_size == 2


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
    assert len(observations) == 1
    clock[0] = 15
    policy.decide(batch)
    assert len(observations) == 2
    clock[0] = 46
    policy.decide(batch)
    assert len(observations) == 3
    assert "future_arrivals" not in observations[0]
    assert "idle_timeout" not in observations[0]
    assert "min_ready_seconds" not in observations[0]


def test_agent_offline_timer_prestarts_and_update_wakes_immediately(
    tmp_path, monkeypatch
):
    """Early start never waits for data; the following update bypasses 40s idle."""
    clock, calls = [0.0], []
    monkeypatch.setattr("areal.elastic.policy.time.monotonic", lambda: clock[0])
    action = ["wait_for_event"]

    def decide(observation):
        calls.append(observation)
        return {
            "action": action[0],
            "reason": "measured progress",
            "wake_after_seconds": 30,
        }

    policy = LifecyclePolicy(mode="agent", decide=decide)
    owner = SimpleNamespace(
        root=tmp_path,
        actor=None,
        options=SimpleNamespace(idle_timeout=40, min_ready_seconds=0),
    )
    status = {
        "generation": 0,
        "applied_version": 0,
        "counts": {},
        "idle_seconds": 0,
        "residence_seconds": 0,
    }
    owner.get_status = lambda **kwargs: dict(status)
    policy.bind(owner)
    assert policy.decide(None) == "keep"
    assert calls[-1]["event"] == "collection_started"
    for _ in range(100):
        assert policy.decide(None) == "keep"
    clock[0] = 30
    action[0] = "start_actor"
    assert policy.decide(None) == "start"
    assert calls[-1]["event"] == "scheduled_wake"
    owner.actor = object()
    status["generation"] = 1
    clock[0] = 200
    assert policy.decide(None) == "keep"  # No start/stop oscillation.
    assert len(calls) == 2
    assert policy.decide(SimpleNamespace(batch_id="b0")) == "train"
    status["applied_version"] = 1
    action[0] = "stop_actor"
    assert policy.decide(None) == "release"  # idle_seconds is still zero.
    assert calls[-1]["event"] == "update_completed"
    owner.actor = None
    action[0] = "wait_for_event"
    assert policy.decide(None) == "keep"
    assert calls[-1]["event"] == "actor_released"
    # A sealed batch bypasses both the wake timer and progress debounce.
    action[0] = "start_actor"
    assert policy.decide(SimpleNamespace(batch_id="b1")) == "train"
    assert calls[-1]["event"] == "batch_ready"


def test_agent_progress_coalesces_elapsed_time_and_key_changes(tmp_path, monkeypatch):
    clock, calls = [0.0], []
    monkeypatch.setattr("areal.elastic.policy.time.monotonic", lambda: clock[0])
    progress = {"event_signature": "rollout", "batch_remaining_seconds": 200}

    def decide(observation):
        calls.append(observation)
        return {"action": "wait_for_event", "reason": "wait", "wake_after_seconds": 60}

    policy = LifecyclePolicy(
        mode="agent", decide=decide, observe=lambda status: dict(progress)
    )
    owner = SimpleNamespace(
        root=tmp_path,
        actor=None,
        options=SimpleNamespace(idle_timeout=40, min_ready_seconds=0),
    )
    owner.get_status = lambda **kwargs: {
        "generation": 1,
        "applied_version": 1,
        "counts": {},
    }
    policy.bind(owner)
    policy.decide(None)
    for second in range(1, 10):
        clock[0] = second
        progress["batch_remaining_seconds"] -= 1
        policy.decide(None)
    assert len(calls) == 1
    progress["event_signature"] = "grading"
    policy.decide(None)
    assert calls[-1]["event"] == "progress_changed"
    progress["event_signature"] = "cleanup"
    policy.decide(None)
    assert len(calls) == 2
    clock[0] += 5
    policy.decide(None)
    assert len(calls) == 3
    assert calls[-1]["workload"]["event_signature"] == "cleanup"


def test_agent_training_does_not_call_llm_for_each_batch(tmp_path):
    calls = []
    policy = LifecyclePolicy(mode="agent", decide=calls.append)
    owner = SimpleNamespace(root=tmp_path, actor=object())
    policy.bind(owner)
    for number in range(20):
        assert policy.decide(SimpleNamespace(batch_id=str(number))) == "train"
    assert calls == []


@pytest.mark.parametrize("running", [False, True])
def test_agent_rejects_wrong_state_action_and_reasks_with_fresh_status(
    tmp_path, running
):
    """A redundant start/stop is diagnosed without terminating the training run."""
    observations = []
    invalid = "start_actor" if running else "stop_actor"
    valid = "stop_actor" if running else "start_actor"
    status = {"generation": 1, "applied_version": 1, "counts": {"ready": 0}}

    def decide(observation):
        observations.append(observation)
        status["idle_seconds"] = 3
        return {
            "action": invalid if len(observations) == 1 else valid,
            "reason": "Use the current state",
        }

    policy = LifecyclePolicy(mode="agent", decide=decide)
    owner = SimpleNamespace(
        root=tmp_path,
        actor=object() if running else None,
        get_status=lambda **kwargs: dict(status),
    )
    policy.bind(owner)

    assert policy.decide(None) == ("release" if running else "start")
    assert len(observations) == 2
    assert observations[1]["event"] == "action_rejected"
    assert observations[1]["rejected_decision"]["action"] == invalid
    assert observations[1]["rejection_reason"]
    assert observations[1]["idle_seconds"] == 3
    assert set(observations[1]["allowed_actions"]) == {valid, "wait_for_event"}
    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text().splitlines()
    ]
    assert [r["outcome"] for r in records] == ["rejected", "accepted"]
    assert records[0]["decision"]["action"] == invalid
    assert records[0]["error"] and records[1]["error"] is None


@pytest.mark.parametrize(
    "proposal",
    [
        None,
        {"action": "unknown", "reason": "bad tool"},
        {"action": "start_actor", "reason": "already online"},
        {"action": "wait_for_event", "reason": ""},
        {"action": "wait_for_event", "reason": "bad timer", "wake_after_seconds": -1},
    ],
)
def test_agent_repeated_invalid_proposal_yields_and_preserves_training_progress(
    tmp_path, monkeypatch, proposal
):
    """Bound correction calls, avoid a polling storm, and let ready data train."""
    clock, calls = [0.0], []
    monkeypatch.setattr("areal.elastic.policy.time.monotonic", lambda: clock[0])

    def decide(observation):
        calls.append(observation)
        clock[0] += 1
        return proposal

    actor = object()
    owner = SimpleNamespace(root=tmp_path, actor=actor)
    owner.get_status = lambda **kwargs: {
        "generation": 1,
        "applied_version": 1,
        "counts": {},
    }
    policy = LifecyclePolicy(mode="agent", decide=decide)
    policy.bind(owner)
    assert policy.decide(None) == "keep"
    assert len(calls) == 2 and owner.actor is actor
    for _ in range(100):
        assert policy.decide(None) == "keep"
    clock[0] = 6.9
    assert policy.decide(None) == "keep"
    assert len(calls) == 2
    assert policy.decide(SimpleNamespace(batch_id="ready")) == "train"
    assert len(calls) == 2
    clock[0] = 7
    assert policy.decide(None) == "keep"
    assert len(calls) == 4
    records = [
        json.loads(line)
        for line in (tmp_path / "decisions.jsonl").read_text().splitlines()
    ]
    assert all(r["outcome"] == "rejected" for r in records)
    assert records[-1]["retry_after_seconds"] == 5


def test_agent_transport_failure_still_propagates(tmp_path):
    """Action validation must not silently replace a failed LLM service."""

    def unavailable(observation):
        raise httpx.ConnectError("test service unavailable")

    owner = SimpleNamespace(root=tmp_path, actor=None)
    owner.get_status = lambda **kwargs: {
        "generation": 0,
        "applied_version": 0,
        "counts": {},
    }
    policy = LifecyclePolicy(mode="agent", decide=unavailable)
    policy.bind(owner)
    with pytest.raises(httpx.ConnectError):
        policy.decide(None)


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
        llm.observe = lambda: {
            "generation": 0,
            "allowed_actions": ["start_actor", "wait_for_event"],
        }
        decision = llm.decide(
            {
                "event": "batch_ready",
                "generation": 0,
                "allowed_actions": ["wait_for_event"],
            }
        )
    assert decision["action"] == "start_actor"
    assert len(requests) == 2
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert "test-only-key" not in json.dumps(requests)
    assert {t["function"]["name"] for t in requests[0]["tools"]} == {
        "get_status",
        "wait_for_event",
    }
    assert {t["function"]["name"] for t in requests[1]["tools"]} == {
        "get_status",
        "start_actor",
        "wait_for_event",
    }


def test_plot_skips_rejected_proposals_and_keeps_legacy_requests(tmp_path):
    """Invalid or non-dict proposals cannot break plots or look like execution."""
    from examples.elastic.plot import plot_run

    (tmp_path / "state").mkdir()
    events = [
        {
            "state": state,
            "monotonic": second,
            "actor_devices": [0, 1, 2, 3],
            "applied_version": 0,
            "counts": {"ready": 0},
        }
        for state, second in [("STARTING", 0), ("STOPPED", 10)]
    ]
    (tmp_path / "state/events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    decisions = [
        {"outcome": "rejected", "decision": None, "monotonic": 1},
        {"outcome": "rejected", "decision": {"action": "stop_actor"}, "monotonic": 2},
        {"decision": {"action": "start_actor"}, "monotonic": 3},
        {
            "outcome": "accepted",
            "decision": {"action": "wait_for_event"},
            "monotonic": 4,
        },
    ]
    (tmp_path / "state/decisions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in decisions)
    )
    svg = plot_run(tmp_path).read_text()
    assert "request start" in svg and "request wait_for_event" in svg
    assert "request stop" not in svg


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


def test_summary_distinguishes_grading_exception_zeros_from_normal_rewards(tmp_path):
    from examples.elastic.run_demo import summarize

    episodes = [
        {"reward": reward, "grading_kind": kind, "turns": 1, "tool_calls": 0}
        for reward, kind in (
            (0.0, "exception_zero"),
            (0.0, "completed_reference_report"),
            (1.0, "completed_reference_report"),
        )
    ]
    (tmp_path / "episodes.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in episodes)
    )
    manager = SimpleNamespace(
        policy_adapter=SimpleNamespace(mode="resident"),
        applied_version=1,
        rollout=SimpleNamespace(serving_versions=lambda: [1, 1]),
        data_progress={"samples": 8},
    )
    result = summarize(tmp_path, manager, 1)
    assert result["completed_workflow_episodes"] == 3
    assert result["completed_workflow_grading_exception_zeros"] == 1
    assert result["data_progress"]["samples"] == 8


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
