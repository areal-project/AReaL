# SPDX-License-Identifier: Apache-2.0

"""Real SDK serialization, model failures and code revision feedback."""

import json
import sys
from pathlib import Path

import httpx
import openai
import pytest

from examples.autopilot.elastic.revision_demo import (
    GOOD_EVALUATOR_SOURCE as AFFINE_SOURCE,
)
from examples.autopilot.elastic.revision_demo import demo_benchmark

from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
from areal.autopilot.domains.elastic.prompt import proposal_context
from areal.autopilot.revision import LLMRevisionAgent, RevisionLoop
from areal.autopilot.revision.contracts import ProposalRequest, digest
from areal.autopilot.revision.execution import CandidateRunner


def model_response(source=AFFINE_SOURCE, **overrides):
    return {
        "id": "test-response",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model-version",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"source": source, "rationale": "Calibration fit"}
                    ),
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        **overrides,
    }


@pytest.fixture
def model_endpoint(monkeypatch):
    """Use the actual SDK against an in-memory HTTP transport, without API spend."""
    requests, replies, configurations = [], [], []
    real_client = openai.OpenAI

    def handle(request):
        requests.append(json.loads(request.content))
        response = replies.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def client(**kwargs):
        configurations.append(kwargs)
        return real_client(
            **kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handle))
        )

    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.setenv("REVISION_TEST_KEY", "test-credential-never-persisted")
    return requests, replies, configurations


def proposer(**kwargs):
    return LLMRevisionAgent(
        model="test-model",
        base_url="http://revision.invalid/v1",
        api_key_env="REVISION_TEST_KEY",
        **kwargs,
    )


def proposal_request():
    profile = demo_benchmark("evaluator").calibration
    return ProposalRequest(
        "evaluator",
        2,
        BASELINE_SOURCE["evaluator"],
        "incumbent-hash",
        profile,
        {},
        ({"status": "rejected", "errors": ["SyntaxError"]},),
        previous_source="broken python !",
        context=proposal_context("evaluator"),
        revision_history=(
            {
                "target": "evaluator",
                "rationale": "Calibrate the train stage",
                "baseline": {"score": 0.3},
                "outcome": {"status": "rejected", "score": 0.3},
            },
        ),
    )


def test_llm_sends_public_api_calibration_and_feedback_and_records_usage(
    model_endpoint,
):
    """A model sees executable API context and only the intended evaluation inputs."""
    requests, replies, configurations = model_endpoint
    replies.append(httpx.Response(200, json=model_response()))
    agent = proposer(reasoning_effort="low", max_output_tokens=2048)
    request = proposal_request()
    result = agent.propose(request)
    sent = requests[0]
    context = json.loads(sent["messages"][1]["content"])
    expected = json.loads(json.dumps(request.to_dict()))
    expected.pop("context")
    assert context["request"] == expected
    assert context["request"]["previous_source"] == "broken python !"
    assert context["request"]["revision_history"][0]["rationale"] == (
        "Calibrate the train stage"
    )
    assert (
        "class HistoricalInput"
        in context["reference_sources"][
            "areal/autopilot/domains/elastic/evaluation/evaluator.py"
        ]
    )
    assert (
        "class CostModelHistoryPredictor"
        in context["reference_sources"][
            "areal/autopilot/domains/elastic/evaluation/evaluator.py"
        ]
    )
    assert "observed" not in context["request"]["calibration"]
    assert "test-credential" not in json.dumps(sent)
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["max_completion_tokens"] == 2048 and sent["reasoning_effort"] == "low"
    assert configurations[0]["max_retries"] == 0
    assert configurations[0]["timeout"] == 180
    assert result.source == AFFINE_SOURCE
    assert result.metadata["usage"]["total_tokens"] == 150
    assert result.metadata["model"] == "test-model-version"
    assert result.metadata["prompt_sha256"] == digest(sent["messages"])
    assert "test-credential" not in json.dumps(result.metadata)


def test_llm_repairs_rejected_source_after_loop_resume(model_endpoint, tmp_path):
    """Persisted failures return both diagnostics and the failed source to the model."""
    requests, replies, _ = model_endpoint
    replies.extend(
        [
            httpx.Response(200, json=model_response("invalid python !")),
            httpx.Response(200, json=model_response()),
        ]
    )
    root = str(Path(__file__).resolve().parents[1])
    bootstrap = f"import sys,types,runpy; p=types.ModuleType('areal'); p.__path__=[{root!r}+'/areal']; sys.modules['areal']=p; runpy.run_module('areal.autopilot.domains.elastic.evaluation.worker',run_name='__main__')"
    evaluation = demo_benchmark(
        "model", CandidateRunner(command=(sys.executable, "-c", bootstrap))
    )
    store = tmp_path / "revision"
    first = RevisionLoop(
        proposer(),
        evaluation,
        store,
        proposal_context=proposal_context(evaluation.target),
    ).run(max_rounds=1)
    assert first["accepted"] == 0 and not (store / "active.json").exists()
    second = RevisionLoop(
        proposer(),
        evaluation,
        store,
        proposal_context=proposal_context(evaluation.target),
    ).run(max_rounds=1)
    assert second["accepted"] == 1
    context = json.loads(requests[1]["messages"][1]["content"])["request"]
    assert context["previous_source"] == "invalid python !"
    assert "SyntaxError" in json.dumps(context["feedback"])
    saved = list((store / "attempts").glob("*/proposal.json"))
    assert all(
        json.loads(path.read_text())["metadata"]["agent"] == "llm" for path in saved
    )


@pytest.mark.parametrize(
    "kind", ["truncated", "refusal", "empty", "bad_json", "extra_fields", "wrong_type"]
)
def test_llm_invalid_output_is_rejected_before_candidate_execution(
    model_endpoint, kind
):
    """Incomplete and malformed model output cannot become executable proposals."""
    _, replies, _ = model_endpoint
    response = model_response()
    choice = response["choices"][0]
    if kind == "truncated":
        choice["finish_reason"] = "length"
    elif kind == "refusal":
        choice["message"]["refusal"] = "refused"
    elif kind == "empty":
        response["choices"] = []
    elif kind == "bad_json":
        choice["message"]["content"] = "```python\ninvalid\n```"
    elif kind == "extra_fields":
        choice["message"]["content"] = json.dumps(
            {"source": "pass", "rationale": "", "score": -1}
        )
    else:
        choice["message"]["content"] = json.dumps({"source": "pass", "rationale": 0})
    replies.append(httpx.Response(200, json=response))
    with pytest.raises(ValueError):
        proposer().propose(proposal_request())


@pytest.mark.parametrize("status", [401, 429, 500])
def test_llm_api_failure_is_bounded_and_does_not_persist_error_body(
    model_endpoint, status
):
    """No hidden SDK retries or endpoint error bodies in saved feedback."""
    requests, replies, _ = model_endpoint
    replies.append(
        httpx.Response(
            status,
            json={
                "error": {
                    "message": "test-credential-never-persisted",
                    "type": "api_error",
                }
            },
        )
    )
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as failure:
        proposer().propose(proposal_request())
    assert "test-credential" not in str(failure.value)
    assert len(requests) == 1


def test_llm_http_timeout_and_explicit_json_mode(model_endpoint):
    """Compatible providers can use JSON mode, with timeouts exposed to the loop."""
    requests, replies, _ = model_endpoint
    replies.append(httpx.ReadTimeout("request timed out"))
    with pytest.raises(TimeoutError, match="timed out"):
        proposer(response_format="json_object").propose(proposal_request())
    assert requests[0]["response_format"] == {"type": "json_object"}


def test_llm_compatible_endpoint_uses_explicit_max_tokens(model_endpoint):
    """OpenRouter-style endpoints receive their documented output token limit."""
    requests, replies, _ = model_endpoint
    replies.append(httpx.Response(200, json=model_response()))
    result = proposer(token_parameter="max_tokens", max_output_tokens=4096).propose(
        proposal_request()
    )
    assert requests[0]["max_tokens"] == 4096
    assert "max_completion_tokens" not in requests[0]
    assert result.metadata["token_parameter"] == "max_tokens"


def test_deepseek_proposal_uses_documented_chat_parameters(model_endpoint, monkeypatch):
    """The existing SDK client sends DeepSeek JSON mode and returns an Optimizer."""
    requests, replies, configurations = model_endpoint
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-test-credential")
    replies.append(
        httpx.Response(
            200,
            json=model_response(BASELINE_SOURCE["optimizer"], model="deepseek-flash"),
        )
    )
    agent = LLMRevisionAgent(
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
        response_format="json_object",
        token_parameter="max_tokens",
        reasoning_effort="none",
    )
    request = ProposalRequest(
        "optimizer",
        1,
        BASELINE_SOURCE["optimizer"],
        "initial-hash",
        {},
        {},
        (),
        context=proposal_context("optimizer"),
    )
    result = agent.propose(request)
    sent = requests[0]
    assert configurations[0]["base_url"] == "https://api.deepseek.com"
    assert sent["model"] == "deepseek-flash"
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["max_tokens"] == 8192 and sent["reasoning_effort"] == "none"
    assert "max_completion_tokens" not in sent
    assert result.source == BASELINE_SOURCE["optimizer"]
    assert "deepseek-test-credential" not in json.dumps(result.metadata)


def test_llm_timeout_is_feedback_and_does_not_abort_revision(model_endpoint, tmp_path):
    """Provider timeouts retry; filesystem failures retain their fail-stop behavior."""
    requests, replies, _ = model_endpoint
    replies.extend(
        [
            httpx.ReadTimeout("request timed out"),
            httpx.Response(200, json=model_response()),
        ]
    )
    root = str(Path(__file__).resolve().parents[1])
    bootstrap = f"import sys,types,runpy; p=types.ModuleType('areal'); p.__path__=[{root!r}+'/areal']; sys.modules['areal']=p; runpy.run_module('areal.autopilot.domains.elastic.evaluation.worker',run_name='__main__')"
    evaluation = demo_benchmark(
        "model", CandidateRunner(command=(sys.executable, "-c", bootstrap))
    )
    result = RevisionLoop(
        proposer(),
        evaluation,
        tmp_path,
        proposal_context=proposal_context(evaluation.target),
    ).run(max_rounds=2)
    assert [item["status"] for item in result["rounds"]] == ["rejected", "accepted"]
    feedback = json.loads(requests[1]["messages"][1]["content"])["request"]["feedback"]
    assert "LLM proposal request timed out" in json.dumps(feedback)


def test_cli_selects_llm_command_without_exposing_api_key(monkeypatch, tmp_path):
    """The user-facing CLI wires the LLM adapter and its wall-clock deadline."""
    import examples.autopilot.revising as cli

    seen = []

    class Loop:
        def __init__(self, agent, evaluation, root, **kwargs):
            seen.append((agent, kwargs))

        def run(self, **kwargs):
            return {"active_source": None, "accepted": 0}

    monkeypatch.setattr(cli, "RevisionLoop", Loop)
    monkeypatch.setattr(cli, "demo_benchmark", lambda *a, **k: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "revising",
            "--target",
            "optimizer",
            "--demo",
            "--output",
            str(tmp_path / "cycle-001"),
            "--history-root",
            str(tmp_path),
            "--llm-model",
            "test-model",
            "--llm-base-url",
            "http://revision.invalid/v1",
            "--llm-api-key-env",
            "REVISION_TEST_KEY",
            "--llm-timeout",
            "20",
        ],
    )
    cli.main()
    agent, options = seen[0]
    assert agent.command[1:3] == ("-m", "areal.autopilot.revision")
    assert "test-model" in agent.command and "REVISION_TEST_KEY" in agent.command
    assert agent.timeout_seconds == 30
    assert "test-credential" not in " ".join(agent.command)
    assert options["proposal_context"]["factory"] == "build_optimizer(settings)"
    assert options["history_root"] == tmp_path


def test_evaluator_proposal_supplies_matching_factory_and_benchmark_api():
    """The existing proposer gets historical predictor APIs without a network call."""
    request = ProposalRequest(
        "evaluator",
        1,
        "def build_cost_models(c, s): pass",
        "source-hash",
        {"cases": []},
        {},
        (),
        context=proposal_context("evaluator"),
    )
    messages = proposer().messages(request)
    data = json.loads(messages[1]["content"])
    assert "build_cost_models" in data["request"]["factory"]
    assert "calibration_cases" in data["request"]["factory"]
    assert (
        "areal/autopilot/domains/elastic/evaluation/evaluator.py"
        in data["reference_sources"]
    )
    assert (
        "final_holdout"
        in data["reference_sources"][
            "areal/autopilot/domains/elastic/evaluation/evaluator.py"
        ]
    )


def test_online_optimizer_proposal_supplies_the_reference_search_port():
    request = ProposalRequest(
        "optimizer",
        1,
        BASELINE_SOURCE["optimizer"],
        "hash",
        {},
        {},
        (),
        context=proposal_context("optimizer"),
    )
    data = json.loads(proposer().messages(request)[1]["content"])
    assert "build_optimizer" in data["request"]["factory"]
    assert (
        "areal/elastic/reconfigurator/reference/optimizer.py"
        in data["reference_sources"]
    )
    assert (
        "SearchPort"
        in data["reference_sources"][
            "areal/autopilot/domains/elastic/evaluation/optimizer.py"
        ]
    )


def test_llm_domain_reference_sources_are_serialized_once():
    from dataclasses import replace

    request = replace(
        proposal_request(),
        context={
            "api": "example()",
            "reference_sources": {"example.py": "UNIQUE_DOMAIN_REFERENCE"},
            "instructions": "UNIQUE_DOMAIN_INSTRUCTION",
        },
    )
    messages = proposer().messages(request)
    assert json.dumps(messages).count("UNIQUE_DOMAIN_REFERENCE") == 1
    assert json.dumps(messages).count("UNIQUE_DOMAIN_INSTRUCTION") == 1
    assert request.to_dict()["context"] == request.context
