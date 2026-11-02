# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys

import pytest

from examples.autopilot.elastic.swe.agent_worker import validate_grading
from examples.autopilot.elastic.swe.process_env import SandboxProcess
from examples.autopilot.elastic.swe.sandbox_worker import call_tool
from examples.autopilot.elastic.swe.trajectory import trajectory_payload


@pytest.fixture
def trajectory():
    return {
        "input_ids": [[1, 2, 3, 0]],
        "attention_mask": [[1, 1, 1, 0]],
        "loss_mask": [[0, 1, 1, 0]],
        "logprobs": [[0.0, -0.1, -0.3, 0.0]],
        "versions": [[-1, 0, 0, -1]],
        "rewards": [0.0],
    }


def payload(trajectory):
    return trajectory_payload(
        trajectory,
        task_id="issue-1",
        rollout_id="session-1",
        model_revision="pinned-revision",
        policy_version=0,
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("versions", [[-1, 1, 0, -1]], "version mismatch"),
        ("versions", [[-1, True, 0, -1]], "version mismatch"),
        ("logprobs", [[0.0, float("nan"), -0.3, 0.0]], "finite"),
        ("attention_mask", [[1, 0, 1, 0]], "mask"),
        ("loss_mask", [[0, 0, 0, 0]], "generated tokens"),
        ("input_ids", [[1, -2, 3, 0]], "Token IDs"),
        ("rewards", [float("inf")], "rewards"),
        ("loss_mask", [[0, 1]], "shape mismatch"),
        ("logprobs", object(), "materialized"),
    ],
)
def test_malformed_or_stale_trajectory_is_rejected(trajectory, field, value, match):
    """Reject data that would silently corrupt a later reconfigurator update."""
    trajectory[field] = value
    with pytest.raises(ValueError, match=match):
        payload(trajectory)


@pytest.mark.parametrize(
    "reward,output,failures",
    [
        (0.0, "SDK error", 0),
        (0.0, "AUTOPILOT_SWE_RESOLVED=true", 0),
        (1.0, "AUTOPILOT_SWE_RESOLVED=true", 1),
        (0.0, "AUTOPILOT_SWE_RESOLVED=true AUTOPILOT_SWE_RESOLVED=false", 0),
        (float("nan"), "AUTOPILOT_SWE_RESOLVED=false", 0),
    ],
)
def test_incomplete_or_infrastructure_failed_grading_is_rejected(
    reward, output, failures
):
    """Do not enqueue swallowed SDK failures as valid negative examples."""
    with pytest.raises(RuntimeError):
        validate_grading(reward, output, failures)


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_completed_negative_and_positive_grading_are_accepted(reward):
    """A real failed solution is valid rollout data when grading completed."""
    validate_grading(reward, "AUTOPILOT_SWE_RESOLVED=" + str(bool(reward)).lower(), 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", ["not-json", '{"id":99,"result":{}}', ""])
async def test_broken_subprocess_protocol_is_fenced_and_child_exits(
    tmp_path, monkeypatch, response
):
    """Malformed replies cannot be reused as a later tool's successful response."""
    create = asyncio.create_subprocess_exec

    async def fake_sdk(*args, **kwargs):
        code = f"import sys; sys.stdin.readline(); print({response!r}, flush=True)"
        return await create(sys.executable, "-u", "-c", code, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_sdk)
    with (tmp_path / "sdk.log").open("w") as diagnostics:
        env = SandboxProcess(sys.executable, stderr=diagnostics)
        try:
            with pytest.raises((ValueError, RuntimeError)):
                await env.start({"instance_id": "test"}, "test@1")
            with pytest.raises(RuntimeError, match="unavailable"):
                await env.execute_bash("pwd")
            assert env.calls == []
        finally:
            await env.close()
        assert env.process.returncode == 0


@pytest.mark.asyncio
async def test_search_replace_preserves_model_schema_with_pinned_sdk_typo(tmp_path):
    """A real tool operation uses the SDK's old_srt without changing model-facing args."""

    class SDK:
        async def search_replace(self, filepath, old_srt, new_str, timeout=100):
            path = tmp_path / filepath
            if old_srt not in path.read_text():
                return {"output": "Text not found", "returncode": 1}
            path.write_text(path.read_text().replace(old_srt, new_str))
            return {"output": "Updated", "returncode": 0}

    path = tmp_path / "source.py"
    path.write_text("old value")
    sdk = SDK()
    args = {"filepath": "source.py", "old_str": "old", "new_str": "new"}
    assert (await call_tool(sdk, "search_replace", args))["returncode"] == 0
    assert path.read_text() == "new value"
    # Ordinary failed edits are tool results, so another valid edit still works.
    assert (await call_tool(sdk, "search_replace", args))["returncode"] == 1
    args.update(old_str="new", new_str="fixed")
    assert (await call_tool(sdk, "search_replace", args))["returncode"] == 0
    assert path.read_text() == "fixed value"
