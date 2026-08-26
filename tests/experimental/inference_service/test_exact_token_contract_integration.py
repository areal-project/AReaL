# SPDX-License-Identifier: Apache-2.0

"""End-to-end proof that a real vLLM enforces the exact-token contract.

The runtime canary probes both directions against whatever servers a run
launches, but only with a synthetic prompt. This exercises the same contract
end-to-end against a real model and real media, including the case the canary
cannot construct: that sending the *expanded* prompt is itself refused.

Run manually. There is no CI job serving a vLLM 0.23 carrying these patches --
the standard vLLM environment pins 0.19.1 -- so marking it must-run would only
fail it against a server that never had the validation installed.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import time
from io import BytesIO

import httpx
import pytest
from PIL import Image

from areal.utils import network
from areal.utils.vision_canary import EXACT_TOKEN_REFUSAL

LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-VL-2B-Instruct/"
HF_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
SERVER_STARTUP_TIMEOUT = 300


def _model_path() -> str:
    """Resolve the test model locally, skipping rather than downloading.

    Pulling a 2B VLM over the network inside a test either hangs or fails as a
    proxy error, and both read as a broken contract when they are really a
    missing prerequisite. Point ``AREAL_TEST_VLM_PATH`` at a local checkout to
    run this on a host that keeps its models elsewhere.
    """
    path = os.environ.get("AREAL_TEST_VLM_PATH", LOCAL_MODEL_PATH)
    if not os.path.isdir(path):
        pytest.skip(
            f"no local copy of {HF_MODEL_ID} at {path}; set AREAL_TEST_VLM_PATH "
            "to run this against a real server"
        )
    return path


def _has_accelerator() -> bool:
    """Whether this host has an accelerator vLLM can serve a VLM on.

    Deliberately not a CUDA check: the exact-token patches ship in the NPU
    images, so gating on CUDA would skip this precisely where it is the only
    coverage. Equally not a bare ``is_available()`` check -- AReaL's CPU
    platform reports available with one device, which would walk a CPU host
    into the server fixture.
    """
    from areal.infra.platforms import current_platform

    try:
        return (
            current_platform.device_type in ("cuda", "npu")
            and current_platform.device_count() > 0
        )
    except Exception:
        return False


def _image_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (112, 112), (127, 127, 127)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture(scope="module")
def vlm_server():
    """Launch a vLLM server carrying the exact-token patches."""
    if not _has_accelerator():
        pytest.skip("no accelerator: this exercises a real vLLM against real media")

    model_path = _model_path()
    host, port = "127.0.0.1", network.find_free_ports(1)[0]
    # Output is inherited rather than piped: a pipe nobody drains fills its
    # buffer and blocks a verbose vLLM part-way through startup.
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "areal.engine.vllm_ext.areal_vllm_server",
            "--model",
            model_path,
            "--host",
            host,
            "--port",
            str(port),
            "--max-model-len",
            "4096",
            "--enforce-eager",
        ],
    )
    base_url = f"http://{host}:{port}"
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                # Not a skip. The hardware prerequisite held, so a server that
                # cannot start is a real failure -- most likely the patches
                # this test exists to check.
                pytest.fail(f"vLLM exited during startup: rc={proc.returncode}")
            try:
                if httpx.get(f"{base_url}/health", timeout=5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(2)
        else:
            pytest.fail(f"vLLM did not become healthy within {SERVER_STARTUP_TIMEOUT}s")
        yield base_url
    finally:
        # vLLM spawns engine-core children; terminating only the parent leaves
        # them holding the accelerator.
        from areal.infra.utils.proc import kill_process_tree

        kill_process_tree(proc.pid)


def _generate(base_url: str, token_ids, expected, image_b64) -> httpx.Response:
    return httpx.post(
        f"{base_url}/inference/v1/generate",
        json={
            "request_id": f"contract-{time.time_ns()}",
            "token_ids": token_ids,
            "expected_token_ids": expected,
            "content_parts": [
                {"type": "image_url", "url": f"data:image/png;base64,{image_b64}"}
            ],
            "sampling_params": {"max_tokens": 1, "temperature": 0.0},
            "stream": False,
        },
        timeout=120,
    )


@pytest.fixture(scope="module")
def prompts():
    """The collapsed and expanded forms of one real multimodal prompt."""
    if not _has_accelerator():
        pytest.skip("no accelerator: this exercises a real vLLM against real media")

    from areal.utils.hf_utils import (
        collapsed_prompt_token_ids,
        load_hf_processor,
    )

    processor = load_hf_processor(_model_path())
    if processor is None:
        pytest.skip("no multimodal processor for the test model")

    text = (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        "Describe this.<|im_end|>\n<|im_start|>assistant\n"
    )
    image = Image.new("RGB", (112, 112), (127, 127, 127))
    collapsed = collapsed_prompt_token_ids(processor, text)
    expanded = list(
        processor(images=[image], text=[text], padding=False)["input_ids"][0]
    )
    return collapsed, expanded


@pytest.mark.slow
@pytest.mark.vllm
@pytest.mark.integration
def test_a_correct_prompt_is_accepted(vlm_server, prompts):
    """Test the positive direction: the server reproduces our expansion.

    This is what proves the two processors agree, and it is the reason the
    collapsed form goes on the wire while the expanded one is asserted.
    """
    collapsed, expanded = prompts

    resp = _generate(vlm_server, collapsed, expanded, _image_b64())

    assert resp.status_code == 200, resp.text


@pytest.mark.slow
@pytest.mark.vllm
@pytest.mark.integration
def test_a_corrupted_expectation_is_refused(vlm_server, prompts):
    """Test the negative direction the runtime canary relies on."""
    collapsed, expanded = prompts
    corrupted = list(expanded)
    corrupted[-1] = 0 if corrupted[-1] != 0 else 1

    resp = _generate(vlm_server, collapsed, corrupted, _image_b64())

    assert resp.status_code == 400, resp.text
    assert EXACT_TOKEN_REFUSAL in resp.text


@pytest.mark.slow
@pytest.mark.vllm
@pytest.mark.integration
def test_sending_the_expanded_prompt_double_expands(vlm_server, prompts):
    """Test that the collapsed form is required, not merely preferred.

    vLLM replaces one placeholder per media item, so an already-expanded prompt
    gains a second run of pad tokens and no longer matches what we would train
    on. The server must refuse it.
    """
    _, expanded = prompts

    resp = _generate(vlm_server, expanded, expanded, _image_b64())

    assert resp.status_code == 400, resp.text
