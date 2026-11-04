# SPDX-License-Identifier: Apache-2.0
"""vLLM sampler evidence, request-local head widths and neutral sampling."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    vLLMConfig,
)
from areal.api.io_struct import ModelRequest
from areal.engine.vllm_remote import VLLMBackend
from areal.utils.vllm_response import parse_vllm_generation_response


def _response(rows, samples, probs=None, reason="length"):
    if probs is None:
        probs = [
            row[f"token_id:{token}"] for row, token in zip(rows, samples, strict=True)
        ]
    return {
        "choices": [
            {
                "finish_reason": reason,
                "logprobs": {
                    "tokens": [f"token_id:{x}" for x in samples],
                    "token_logprobs": probs,
                    "top_logprobs": rows,
                },
            }
        ]
    }


@pytest.mark.parametrize("width", [1, 2, 128])
def test_vllm_recovers_fixed_head_inside_outside_and_tied_sample(width):
    head = {f"token_id:{i}": -7.0 for i in range(width)}
    rows = [head, {"token_id:999": -7.0, **head}]
    actual = parse_vllm_generation_response(
        _response(rows, [0, 999]), score_centering_top_k=width
    )
    assert actual.output_tokens == [0, 999]
    assert actual.sampler_head_ids == [list(range(width))] * 2
    assert actual.sampler_head_logprobs == [[-7.0] * width] * 2
    # In particular the tied, sampled tail token never enters the fixed head.
    assert 999 not in actual.sampler_head_ids[1]


@pytest.mark.parametrize(
    "row,sample,prob",
    [
        (None, 0, -2.0),
        ({}, 0, -2.0),
        ({"token_id:0": -2.0}, 0, -2.0),
        ({f"token_id:{i}": -3.0 for i in range(4)}, 0, -3.0),
        ({"decoded text": -2.0, "token_id:1": -3.0}, 0, -2.0),
        ({"token_id:-1": -2.0, "token_id:1": -3.0}, 0, -2.0),
        ({"token_id:0:1": -2.0, "token_id:1": -3.0}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:00": -2.0}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:1": None}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:1": True}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:1": "-3"}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:1": float("nan")}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:1": float("-inf")}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:1": float("inf")}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:1": 0.1}, 0, -2.0),
        ({"token_id:0": 0.0, "token_id:1": 0.0}, 0, 0.0),
        ({"token_id:0": -2.0, "token_id:1": -9999.0}, 0, -2.0),
        ({"token_id:0": -2.0, "token_id:1": -3.0}, 99, -4.0),
        ({"token_id:0": -2.0, "token_id:1": -3.0, "token_id:2": -4.0}, 99, -4.0),
        ({"token_id:0": -2.0, "token_id:1": -3.0}, 0, -2.1),
        ({"token_id:0": -2.0, "token_id:1": -3.0}, 0, None),
    ],
)
def test_vllm_rejects_invalid_sampler_evidence(row, sample, prob):
    with pytest.raises(ValueError):
        parse_vllm_generation_response(
            _response([row], [sample], [prob]), score_centering_top_k=2
        )


@pytest.mark.parametrize("rows", [None, [], [{"token_id:0": -2.0}] * 2])
def test_vllm_rejects_missing_or_misaligned_head_rows(rows):
    with pytest.raises(ValueError):
        parse_vllm_generation_response(
            _response(rows, [0], [-2.0]), score_centering_top_k=1
        )


def test_vllm_disabled_heads_do_not_change_existing_response_parsing():
    result = parse_vllm_generation_response(_response([None], [0], [-2.0]))
    assert result.output_logprobs == [-2.0]
    assert result.sampler_head_ids is None
    assert result.sampler_head_logprobs is None


def test_vllm_rejects_sampled_tail_that_outranks_returned_head():
    response = _response(
        [{"token_id:9": -2.0, "token_id:1": -3.0, "token_id:2": -4.0}], [9]
    )
    with pytest.raises(ValueError, match="outranks"):
        parse_vllm_generation_response(response, score_centering_top_k=2)


@pytest.mark.parametrize("tail_logprob", [-3.0, -3.0 + 1e-6])
def test_vllm_accepts_sampled_tail_tied_with_head_boundary(tail_logprob):
    response = _response(
        [{"token_id:9": tail_logprob, "token_id:1": -2.0, "token_id:2": -3.0}], [9]
    )
    result = parse_vllm_generation_response(response, score_centering_top_k=2)
    assert result.sampler_head_ids == [[1, 2]]
    assert result.sampler_head_logprobs == [[-2.0, -3.0]]


def test_vllm_empty_abort_needs_no_heads():
    result = parse_vllm_generation_response(
        _response([], [], reason="abort"), score_centering_top_k=128
    )
    assert result.output_tokens == []
    assert result.stop_reason == "abort"


def test_vllm_request_neutralizes_inherited_generation_defaults():
    req = ModelRequest(
        input_ids=[1, 2], gconfig=GenerationHyperparameters(score_centering_top_k=128)
    )
    payload = VLLMBackend().build_generation_request(req, False, 0).payload
    assert {
        k: payload[k]
        for k in [
            "logprobs",
            "top_k",
            "top_p",
            "temperature",
            "min_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "min_tokens",
        ]
    } == dict(
        logprobs=128,
        top_k=-1,
        top_p=1.0,
        temperature=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        min_tokens=0,
    )
    assert payload["return_tokens_as_token_ids"] is True


@pytest.mark.parametrize(
    "options",
    [
        dict(top_k=10),
        dict(top_p=0.9),
        dict(temperature=0.8),
        dict(greedy=True),
        dict(frequency_penalty=1.0),
        dict(use_beam_search=True),
    ],
)
def test_vllm_rejects_transformed_sampling(options):
    req = ModelRequest(
        gconfig=GenerationHyperparameters(score_centering_top_k=2, **options)
    )
    with pytest.raises(ValueError):
        VLLMBackend().build_generation_request(req, False, 0)


def test_vllm_rejects_vision_and_v2_score_centering():
    from areal.v2.inference_service.vllm.bridge import VLLMBridgeBackend

    req = ModelRequest(gconfig=GenerationHyperparameters(score_centering_top_k=2))
    with pytest.raises(ValueError, match="v1"):
        VLLMBridgeBackend().build_generation_request(req, False)
    req.vision_msg_vllm = [[{"role": "user", "content": "test"}]]
    with pytest.raises(ValueError, match="text completions"):
        VLLMBackend().build_generation_request(req, False, 0)


@pytest.mark.parametrize("width", [0, 1, 20, 128, 1024])
def test_vllm_server_args_raise_cap_only_for_score_centering(width):
    config = vLLMConfig(model="local-model")
    before = copy.deepcopy(config)
    args = vLLMConfig.build_args(config, 1, 1, score_centering_top_k=width)
    assert config == before
    if width:
        assert args["max_logprobs"] == max(20, width)
        assert args["logprobs_mode"] == "raw_logprobs"
        assert args["generation_config"] == "vllm"
        cmd = vLLMConfig.build_cmd(config, 1, 1, score_centering_top_k=width)
        assert cmd[cmd.index("--max-logprobs") + 1] == str(max(20, width))
        assert cmd[cmd.index("--generation-config") + 1] == "vllm"
    else:
        assert not {"max_logprobs", "logprobs_mode", "generation_config"} & args.keys()


@pytest.mark.parametrize("width", [-1, 1025, True, 1.5, "128"])
def test_vllm_server_args_reject_invalid_head_width(width):
    with pytest.raises(ValueError, match="head width"):
        vLLMConfig.build_args(vLLMConfig(), 1, 1, score_centering_top_k=width)


@pytest.mark.asyncio
async def test_vllm_interruption_keeps_request_local_heads_and_versions(monkeypatch):
    from areal.infra import remote_inf_engine as module
    from areal.infra.remote_inf_engine import RemoteInfEngine

    engine = RemoteInfEngine(InferenceEngineConfig(), backend=VLLMBackend())
    engine.choose_server = lambda: "test-server"
    engine.workflow_executor = SimpleNamespace(is_paused=lambda: False)
    versions = iter([7, 8, 9])
    engine.get_version = lambda: next(versions)
    responses = [
        _response([], [], reason="abort"),
        _response(
            [{"token_id:9": -4.0, "token_id:1": -2.0, "token_id:2": -3.0}],
            [9],
            reason="abort",
        ),
        _response([{"token_id:3": -2.0, "token_id:4": -3.0}], [3]),
    ]
    monkeypatch.setattr(
        module.workflow_context, "get_aiohttp_session", AsyncMock(return_value=None)
    )
    calls = []

    async def request(**kwargs):
        calls.append(copy.deepcopy(kwargs["payload"]))
        return responses.pop(0)

    monkeypatch.setattr(module, "arequest_with_retry", request)
    req = ModelRequest(
        input_ids=[10, 11],
        gconfig=GenerationHyperparameters(
            max_new_tokens=2, top_k=-1, score_centering_top_k=2
        ),
    )
    result = await engine.agenerate(req)
    assert result.output_tokens == [9, 3]
    assert result.output_logprobs == [-4.0, -2.0]
    assert result.sampler_head_ids == [[1, 2], [3, 4]]
    assert result.sampler_head_logprobs == [[-2.0, -3.0]] * 2
    assert result.output_versions == [8, 9]
    assert [p["logprobs"] for p in calls] == [2, 2, 2]
    assert calls[-1]["prompt"] == [10, 11, 9]


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [0, 2, 128])
async def test_vllm_routed_interruption_preserves_heads_and_lease_versions(
    monkeypatch, width
):
    from areal.infra import remote_inf_engine as module
    from areal.infra.remote_inf_engine import RemoteInfEngine

    backend = VLLMBackend()
    parse = Mock(wraps=backend.parse_generation_response)
    monkeypatch.setattr(backend, "parse_generation_response", parse)
    engine = RemoteInfEngine(InferenceEngineConfig(), backend=backend)
    engine.workflow_executor = SimpleNamespace(is_paused=lambda: False)
    engine.get_version = lambda: 100  # The serving lease, not this value, owns tokens.
    head = {f"token_id:{i}": -7.0 for i in range(max(width, 1))}
    responses = iter(
        [
            _response([head], [0], reason="abort"),
            _response([head], [0]),
        ]
    )

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        async def json(self):
            return next(responses)

    session = SimpleNamespace(
        request=Mock(side_effect=lambda *args, **kwargs: Response())
    )
    monkeypatch.setattr(
        module.workflow_context, "get_aiohttp_session", AsyncMock(return_value=session)
    )
    versions = iter([7, 9])
    terminal_states = []

    async def execute(actual_session, session_id, generate):
        assert actual_session is session and session_id == "test-session"
        version = next(versions)
        result, terminal = await generate(
            SimpleNamespace(backend_addr="http://test-server", weight_version=version)
        )
        terminal_states.append(terminal)
        return result, version

    engine.route_client = SimpleNamespace(execute=execute)
    request = ModelRequest(
        input_ids=[10, 11],
        gconfig=GenerationHyperparameters(
            max_new_tokens=2, score_centering_top_k=width
        ),
    )
    context = module.current_backend_session.set("test-session")
    try:
        result = await engine.agenerate(request)
    finally:
        module.current_backend_session.reset(context)

    assert result.output_tokens == [0, 0]
    assert result.output_versions == [7, 9]
    assert terminal_states == ["aborted", "completed"]
    assert session.request.call_args.kwargs["json"]["prompt"] == [10, 11, 0]
    assert [call.kwargs for call in parse.call_args_list] == [
        {"score_centering_top_k": width} if width else {}
    ] * 2
    if width:
        assert result.sampler_head_ids == [list(range(width))] * 2
        assert result.sampler_head_logprobs == [[-7.0] * width] * 2
    else:
        assert result.sampler_head_ids is None
        assert result.sampler_head_logprobs is None


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
async def test_vllm_concurrent_requests_keep_independent_head_widths(
    monkeypatch, routed
):
    from areal.infra import remote_inf_engine as module
    from areal.infra.remote_inf_engine import RemoteInfEngine

    engine = RemoteInfEngine(InferenceEngineConfig(), backend=VLLMBackend())
    engine.choose_server = lambda: "test-server"
    engine.workflow_executor = SimpleNamespace(is_paused=lambda: False)
    engine.get_version = lambda: 7
    monkeypatch.setattr(
        module.workflow_context, "get_aiohttp_session", AsyncMock(return_value=None)
    )
    both_started = asyncio.Event()
    calls = []

    async def request(**kwargs):
        width = kwargs["payload"]["logprobs"]
        calls.append(width)
        if len(calls) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=5)
        return _response([{f"token_id:{i}": -7.0 for i in range(width)}], [0])

    monkeypatch.setattr(module, "arequest_with_retry", request)

    if routed:

        class Response:
            def __init__(self, payload):
                self.payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def raise_for_status(self):
                pass

            async def json(self):
                return await request(payload=self.payload)

        session = SimpleNamespace(
            request=lambda *args, **kwargs: Response(kwargs["json"])
        )
        monkeypatch.setattr(
            module.workflow_context,
            "get_aiohttp_session",
            AsyncMock(return_value=session),
        )

        async def execute(actual_session, session_id, generate):
            result, terminal = await generate(
                SimpleNamespace(backend_addr="http://test-server", weight_version=7)
            )
            assert terminal == "completed"
            return result, 7

        engine.route_client = SimpleNamespace(execute=execute)

    context = module.current_backend_session.set("test-session")
    try:
        results = await asyncio.gather(
            *[
                engine.agenerate(
                    ModelRequest(
                        input_ids=[10],
                        gconfig=GenerationHyperparameters(
                            max_new_tokens=1, score_centering_top_k=width
                        ),
                    )
                )
                for width in (2, 128)
            ]
        )
    finally:
        module.current_backend_session.reset(context)
    assert [len(result.sampler_head_ids[0]) for result in results] == [2, 128]


@pytest.mark.parametrize(
    "token", ["decoded text", "token:0", "token_id:-1", "token_id:0:1", 0]
)
def test_vllm_score_centering_rejects_malformed_sample_encoding(token):
    response = _response([{"token_id:0": -2.0}], [0])
    response["choices"][0]["logprobs"]["tokens"] = [token]
    with pytest.raises(ValueError, match="encoding"):
        parse_vllm_generation_response(response, score_centering_top_k=1)


@pytest.mark.parametrize(
    "train_width,eval_width", [(128, 2), (2, 128), (0, 128), (0, 0)]
)
@pytest.mark.parametrize("is_eval", [False, True])
def test_vllm_trainer_passes_shared_server_head_cap(
    monkeypatch, train_width, eval_width, is_eval
):
    from areal.trainer import rl_trainer as module

    config = SimpleNamespace(
        rollout=InferenceEngineConfig(backend="vllm:d1"),
        vllm=vLLMConfig(model="test-model"),
        gconfig=GenerationHyperparameters(score_centering_top_k=train_width),
        eval_gconfig=GenerationHyperparameters(score_centering_top_k=eval_width),
    )
    controller = Mock()
    monkeypatch.setattr(module, "is_single_controller", lambda: True)
    monkeypatch.setattr(
        module.RemotevLLMEngine, "as_controller", Mock(return_value=controller)
    )
    trainer = SimpleNamespace(
        config=config,
        rollout_alloc=SimpleNamespace(
            backend="vllm", parallel=SimpleNamespace(tp_size=1, pp_size=1)
        ),
        scheduler=None,
        mopd_execution_plan=None,
    )
    trainer.rollout = SimpleNamespace(server_infos=["existing-server"])
    module.PPOTrainer._init_rollout(trainer, config.rollout, is_eval=is_eval)
    if is_eval:
        assert controller.initialize.call_args.kwargs["server_infos"] == [
            "existing-server"
        ]
        assert controller.initialize.call_args.kwargs["role"] == "eval-rollout"
    args = controller.initialize.call_args.kwargs["server_args"]
    if max(train_width, eval_width):
        assert args["max_logprobs"] == max(20, train_width, eval_width)
        assert args["logprobs_mode"] == "raw_logprobs"
        assert args["generation_config"] == "vllm"
    else:
        assert "max_logprobs" not in args


@pytest.mark.parametrize(
    "train_width,eval_width", [(128, 2), (2, 128), (0, 128), (0, 0)]
)
def test_vllm_legacy_launcher_forwards_shared_server_head_cap(
    monkeypatch, train_width, eval_width
):
    from areal.infra.launcher import vllm_server as module

    config = SimpleNamespace(
        vllm=vLLMConfig(),
        cluster=SimpleNamespace(name_resolve=None, n_gpus_per_node=2),
        rollout=InferenceEngineConfig(),
        allocation_mode="vllm:d1+fsdp:d1",
        experiment_name="test",
        trial_name="test",
        gconfig=GenerationHyperparameters(score_centering_top_k=train_width),
        eval_gconfig=GenerationHyperparameters(score_centering_top_k=eval_width),
    )
    wrapper = Mock()
    monkeypatch.setattr(module, "parse_cli_args", lambda argv: (config, None))
    monkeypatch.setattr(module, "to_structured_cfg", lambda value, cls: value)
    monkeypatch.setattr(module.name_resolve, "reconfigure", Mock())
    monkeypatch.setattr(
        module._AllocationMode,
        "from_str",
        lambda value: SimpleNamespace(gen_backend="vllm"),
    )
    monkeypatch.setattr(
        module, "get_scheduling_spec", lambda value: SimpleNamespace(cpu=2)
    )
    monkeypatch.setattr(module, "vLLMServerWrapper", wrapper)
    with pytest.warns(FutureWarning):
        module.launch_vllm_server([])
    assert wrapper.call_args.kwargs["score_centering_top_k"] == max(
        train_width, eval_width
    )
    wrapper.return_value.run.assert_called_once()


def test_vllm_legacy_wrapper_passes_head_cap_to_server_command(monkeypatch):
    from areal.infra.launcher import vllm_server as module

    monkeypatch.setattr(
        module,
        "current_platform",
        SimpleNamespace(device_control_env_var="AREAL_TEST_VISIBLE_DEVICES"),
    )
    monkeypatch.setattr(module.signal, "signal", Mock())
    monkeypatch.setattr(module, "find_free_ports", lambda *args: (18000, 18001))
    monkeypatch.setattr(module, "gethostip", lambda: "127.0.0.1")
    monkeypatch.setenv(module.current_platform.device_control_env_var, "0")
    allocation = SimpleNamespace(
        gen_instance_size=1, gen=SimpleNamespace(tp_size=1, pp_size=1)
    )
    wrapper = module.vLLMServerWrapper(
        "test",
        "test",
        vLLMConfig(model="test-model"),
        allocation,
        1,
        score_centering_top_k=128,
    )
    wrapper._shutdown_requested = True
    launch = Mock(return_value=Mock(poll=lambda: 0))
    monkeypatch.setattr(wrapper, "launch_one_server", launch)
    wrapper.run()
    cmd = launch.call_args.args[0]
    assert cmd[cmd.index("--max-logprobs") + 1] == "128"
    assert cmd[cmd.index("--logprobs-mode") + 1] == "raw_logprobs"
    assert cmd[cmd.index("--generation-config") + 1] == "vllm"


@pytest.mark.parametrize("width", [-1, 1025, True, 1.5, "128"])
def test_vllm_parser_rejects_invalid_width_even_on_empty_response(width):
    with pytest.raises(ValueError, match="head width"):
        parse_vllm_generation_response(
            _response([], [], reason="abort"), score_centering_top_k=width
        )


def test_vllm_parser_rejects_positive_sample_logprob_within_match_tolerance():
    with pytest.raises(ValueError):
        parse_vllm_generation_response(
            _response([{"token_id:0": 0.0}], [0], [1e-6]), score_centering_top_k=1
        )


@pytest.mark.parametrize("train_width,eval_width", [(1, 0), (0, 1)])
def test_vllm_v2_config_rejects_training_or_eval_score_centering(
    train_width, eval_width
):
    from areal.api.cli_args import PPOConfig

    with pytest.raises(ValueError, match="v1"):
        PPOConfig(
            rollout=InferenceEngineConfig(backend="vllm:d1", _version="v2"),
            gconfig=GenerationHyperparameters(score_centering_top_k=train_width),
            eval_gconfig=GenerationHyperparameters(score_centering_top_k=eval_width),
        )


@pytest.mark.slow
@pytest.mark.parametrize("tis_cap", [None, 2.0])
def test_vllm_local_checkpoint_probabilities_and_optimizer(tis_cap):
    import os
    import subprocess
    import sys
    from pathlib import Path

    import torch

    model, url = os.getenv("AREAL_TEST_MODEL"), os.getenv("AREAL_TEST_VLLM_URL")
    if not model or not url:
        pytest.skip(
            "Set AREAL_TEST_MODEL and AREAL_TEST_VLLM_URL for a live vLLM server."
        )
    if not torch.cuda.is_available():
        pytest.skip("A separate CUDA device is required for the training process.")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=1",
        str(Path(__file__).parent / "torchrun" / "run_vllm_score_centering.py"),
        "--model",
        model,
        "--server-url",
        url,
    ]
    if tis_cap is not None:
        command.extend(["--tis-cap", str(tis_cap)])
    subprocess.run(command, check=True, timeout=600)
