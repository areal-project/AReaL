import os
import re
import subprocess
import time

import pytest

from tests.test_examples import run_example
from tests.utils import get_model_path

from areal.infra.platforms import current_platform
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging

logger = logging.getLogger("InferenceServiceExamples")

pytestmark = pytest.mark.slow


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_tau2_rollout(tmp_path_factory):
    tau2 = pytest.importorskip("tau2")
    del tau2

    tau2_data_dir = os.environ.get("TAU2_DATA_DIR")
    if not tau2_data_dir:
        pytest.skip("TAU2_DATA_DIR environment variable not set. Skipping tau2 test.")
    if not os.path.exists(tau2_data_dir):
        pytest.skip(
            f"TAU2_DATA_DIR ({tau2_data_dir}) does not exist. Skipping tau2 test."
        )

    chat_template_path = "/storage/openpsi/data/qwen3_nonthinking.jinja"
    if not os.path.exists(chat_template_path):
        pytest.skip(f"Chat template not found at {chat_template_path}")

    if current_platform.device_count() < 3:
        pytest.skip(
            "This test requires at least 3 GPUs (1 for user LLM, 2 for rollout) to run."
        )

    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )

    visible_devices = os.getenv(
        current_platform.device_control_env_var,
        ",".join(map(str, range(current_platform.device_count()))),
    ).split(",")
    assert len(visible_devices) >= 3

    user_llm_gpu = visible_devices[-1]
    user_llm_port = 30081

    _env = os.environ.copy()
    _env[current_platform.device_control_env_var] = user_llm_gpu

    logger.info(
        f"Launching user LLM server on GPU {user_llm_gpu}, port {user_llm_port}"
    )
    user_llm_proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(user_llm_port),
            "--tool-call-parser",
            "qwen25",
            "--chat-template",
            chat_template_path,
            "--dp-size",
            "1",
            "--mem-fraction-static",
            "0.8",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_env,
    )

    try:
        logger.info("Waiting for user LLM server to start...")
        time.sleep(60)

        user_llm_base_url = f"http://localhost:{user_llm_port}/v1/"
        success = run_async_task(
            run_example,
            "examples/experimental/inference_service/tau2_rollout.py",
            "examples/experimental/inference_service/tau2_rollout.yaml",
            "rollout.backend=sglang:d2",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            "rollout.admin_api_key=test-admin-key",
            f"model_path={model_path}",
            "train_dataset.batch_size=2",
            "train_dataset.path=tau2/train",
            f"econfig.user_llm_base_url={user_llm_base_url}",
            "econfig.user_llm=openai/self-hosted-qwen3",
            "stats_logger.wandb.mode=disabled",
            timeout=600,
            success_pattern=re.compile(r"Rollout complete"),
        )
        assert success, "Tau2 rollout example failed"
    finally:
        logger.info("Shutting down user LLM server...")
        kill_process_tree(user_llm_proc.pid, graceful=False)
