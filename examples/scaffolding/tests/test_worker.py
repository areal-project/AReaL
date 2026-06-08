"""Tests for SGLangWorker with a real SGLang server (requires GPU)."""

from __future__ import annotations

import subprocess
import sys
import time
from unittest.mock import MagicMock

import openai
import pytest
import requests

from examples.scaffolding._compat import (
    GenerationTask,
    TaskStatus,
)
from examples.scaffolding.worker import SGLangWorker

from areal.api.cli_args import SGLangConfig
from areal.tests.utils import get_model_path
from areal.utils import network, seeding
from areal.utils.hf_utils import apply_chat_template, load_hf_tokenizer
from areal.utils.proc import kill_process_tree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPR_NAME = "test_scaffolding_worker"
MODEL_PATH = get_model_path(
    "/storage/openpsi/models/Qwen__Qwen3-0.6B/", "Qwen/Qwen3-0.6B"
)
PORT, DIST_PORT = network.find_free_ports(2)
HOST = network.gethostip()
RUN_SERVER_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _check_server_health(base_url: str) -> bool:
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="module")
def sglang_server():
    """Launch a real SGLang server.

    Uses skip_tokenizer_init=True (AReaL default). Chat template is applied
    client-side via the tokenizer, matching ScaffoldingWorkflow behavior.
    """
    seeding.set_random_seed(1, EXPR_NAME)
    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=False,
            model_path=MODEL_PATH,
            mem_fraction_static=0.3,
        ),
        host=HOST,
        port=PORT,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{HOST}:{DIST_PORT}",
    )
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stdout,
    )
    base_url = f"http://{HOST}:{PORT}"
    tik = time.time()
    while time.time() - tik < RUN_SERVER_TIMEOUT:
        if _check_server_health(base_url):
            break
        time.sleep(1)
    if time.time() - tik > RUN_SERVER_TIMEOUT:
        kill_process_tree(process.pid, graceful=True)
        raise RuntimeError("SGLang server launch timed out")
    yield base_url
    kill_process_tree(process.pid, graceful=True)


@pytest.fixture(scope="module")
def tokenizer():
    """Load the tokenizer for client-side chat template application."""
    return load_hf_tokenizer(MODEL_PATH)


@pytest.fixture(scope="module")
def sglang_worker(sglang_server):
    """Create an SGLangWorker connected to the real SGLang server."""
    base_url = sglang_server
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    async_client = openai.AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    mock_engine = MagicMock()
    return SGLangWorker(
        async_client=async_client,
        model="default",
        engine=mock_engine,
    )


# ---------------------------------------------------------------------------
# Tests — generation_handler (uses /v1/completions, no tokenizer needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_handler(sglang_worker):
    """generation_handler should generate text from real server."""
    task = GenerationTask(
        input_str="What is 1 + 1? Answer briefly:",
        input_tokens=[],
    )
    status = await sglang_worker.generation_handler(task)

    assert status == TaskStatus.SUCCESS
    assert task.output_str is not None
    assert len(task.output_str) > 0
    assert task.finish_reason in ("stop", "length")


@pytest.mark.asyncio
async def test_generation_max_tokens(sglang_worker):
    """generation_handler should respect max_tokens and finish with 'length'."""
    task = GenerationTask(
        input_str="Write a very long essay about the history of mathematics.",
        input_tokens=[],
    )
    original_create = sglang_worker.async_client.completions.create

    async def _create_with_limit(**kwargs):
        kwargs["max_tokens"] = 5
        return await original_create(**kwargs)

    sglang_worker.async_client.completions.create = _create_with_limit
    try:
        status = await sglang_worker.generation_handler(task)
        assert status == TaskStatus.SUCCESS
        assert task.finish_reason == "length"
        assert task.output_str is not None
    finally:
        sglang_worker.async_client.completions.create = original_create


# ---------------------------------------------------------------------------
# Tests — generation with chat template (client-side, like ScaffoldingWorkflow)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_with_chat_template(sglang_worker, tokenizer):
    """Client-side chat template + completions API, matching ScaffoldingWorkflow."""
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    input_ids = apply_chat_template(
        tokenizer, messages, tokenize=True, add_generation_prompt=True
    )
    prompt_str = tokenizer.decode(input_ids)

    task = GenerationTask(
        input_str=prompt_str,
        input_tokens=input_ids,
    )
    status = await sglang_worker.generation_handler(task)

    assert status == TaskStatus.SUCCESS
    assert task.output_str is not None
    assert len(task.output_str) > 0


@pytest.mark.asyncio
async def test_multi_turn_generation(sglang_worker, tokenizer):
    """Multi-turn via client-side chat template + completions API."""
    # Turn 1
    messages = [{"role": "user", "content": "My name is Alice."}]
    input_ids = apply_chat_template(
        tokenizer, messages, tokenize=True, add_generation_prompt=True
    )
    prompt_str = tokenizer.decode(input_ids)

    task = GenerationTask(input_str=prompt_str, input_tokens=input_ids)
    status = await sglang_worker.generation_handler(task)
    assert status == TaskStatus.SUCCESS
    assert task.output_str is not None

    # Turn 2 — append assistant reply and new user message
    messages.append({"role": "assistant", "content": task.output_str})
    messages.append({"role": "user", "content": "What is my name?"})
    input_ids_2 = apply_chat_template(
        tokenizer, messages, tokenize=True, add_generation_prompt=True
    )
    prompt_str_2 = tokenizer.decode(input_ids_2)

    task2 = GenerationTask(input_str=prompt_str_2, input_tokens=input_ids_2)
    status = await sglang_worker.generation_handler(task2)
    assert status == TaskStatus.SUCCESS
    assert task2.output_str is not None
