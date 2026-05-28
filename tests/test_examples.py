import asyncio
import os
import re
import shutil
import signal
import subprocess
import time
import uuid

import pytest

from tests.utils import get_dataset_path, get_model_path

from areal.infra.platforms import current_platform
from areal.infra.utils.concurrent import run_async_task
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging

logger = logging.getLogger("TestExamples")

SUCCESS_PATTERN = re.compile(r"Epoch 1/\d+ Step 1/\d+ Train step 1/\d+ done\.")

pytestmark = pytest.mark.slow


async def run_example(
    example_file: str,
    config_name: str,
    *additional_args,
    timeout: int = 480,
    success_pattern=SUCCESS_PATTERN,
) -> bool:
    """
    Run a single example in single-controller mode and return the result.

    Args:
        example_file: Path to the example file
        config_name: Name of the config to use
        additional_args: Additional command line arguments
        timeout: Timeout in seconds
        success_pattern: Regex pattern to identify successful completion

    Returns:
        True if the success pattern was found in stdout, False otherwise.
    """
    # Construct the command (single-controller mode: run script directly)
    cmd = [
        "python3",
        example_file,
        "--config",
        config_name,
    ]
    cmd += list(additional_args)

    logger.info(f"Running: {' '.join(cmd)}")

    # Run the command with timeout
    success = False
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    start_time = time.monotonic()

    while True:
        # Read output by line
        while True:
            try:
                line = await asyncio.wait_for(process.stdout.readline(), timeout=0.1)
                line = line.decode().rstrip()
                # Skip empty lines (e.g., from tqdm progress bar cleanup)
                if line:
                    logger.info(f"[Example Output] {line}")
                # Check for success patterns
                success = bool(success_pattern.search(line))
                if success:
                    break
            except (TimeoutError, ValueError):
                # NOTE: Here ValueError is raised when the input line is too long
                # that exceeds the buffer size, which will happen if the experiment
                # has tqdm progress bar output.
                break

        if success:
            logger.info(f"✓ {example_file} with config {config_name} - SUCCESS")
            process.send_signal(signal.SIGINT)  # Gracefully terminate the process
            break

        # Check if process has terminated
        try:
            return_code = await asyncio.wait_for(process.wait(), timeout=0.01)
            logger.error(f"Process terminated unexpectedly. Return code: {return_code}")
            break
        except TimeoutError:
            pass

        # Check timeout
        if (time.monotonic() - start_time) > timeout:
            logger.error("Process timed out without successful result, terminating...")
            process.send_signal(signal.SIGINT)  # Gracefully terminate the process
            break

    kill_process_tree(process.pid)
    return success


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_countdown_example(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    tmp_path = tmp_path_factory.mktemp("countdown_data")
    data_path = tmp_path / "data/countdown/qwen"
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    os.makedirs(data_path, exist_ok=True)
    test_file_path = data_path / "test_e.jsonl"
    train_file_path = data_path / "train_e.jsonl"
    # generate countdown dataset
    shutil.copy("examples/countdown/countdown.py", tmp_path)
    subprocess.run(
        [
            "python3",
            "countdown.py",
            "--num_samples=10000",
            "--eval_size=100",
            "--tokenizer_path",
            model_path,
        ],
        cwd=tmp_path,
        check=True,
    )

    example_file = "examples/countdown/train.py"
    config_name = "examples/countdown/train_config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "actor.backend=fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=128",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={str(train_file_path)}",
        f"valid_dataset.path={str(test_file_path)}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
    )
    assert success, "Countdown example failed"


# vLLM is too slow to launch up in CI environments
# We have tests for vLLM in test_inference_engines.py,
# so we can skip the integration test of vLLM here.
@pytest.mark.sglang
@pytest.mark.multi_gpu
@pytest.mark.ci
def test_gsm8k_grpo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_rl.py"
    config_name = "examples/math/gsm8k_grpo.yaml"

    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "actor.backend=megatron:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=1",
        "valid_dataset.batch_size=1",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        timeout=900,
    )
    assert success, "GSM8K GRPO example failed"


@pytest.mark.parametrize(
    "actor_backend",
    [
        "fsdp:d1",
        "megatron:d1",
    ],
)
@pytest.mark.gpu
@pytest.mark.ci
def test_gsm8k_sft(tmp_path_factory, actor_backend):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_sft.py"
    config_name = "examples/math/gsm8k_sft.yaml"

    success = run_async_task(
        run_example,
        example_file,
        config_name,
        f"actor.backend={actor_backend}",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=1",
        "valid_dataset.batch_size=1",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
    )
    assert success, f"GSM8K SFT example failed (actor_backend={actor_backend})"


@pytest.mark.sglang
@pytest.mark.gpu
def test_gsm8k_eval(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_eval.py"
    config_name = "examples/math/gsm8k_grpo.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "gconfig.n_samples=1",
        "gconfig.max_new_tokens=16",
        "valid_dataset.batch_size=16",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        success_pattern=re.compile(r"Evaluation Results:"),
    )
    assert success, "GSM8K Eval example failed"


@pytest.mark.ci
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "rollout_backend,actor_backend",
    [
        pytest.param("sglang:d1", "fsdp:d1", marks=pytest.mark.sglang),
        pytest.param("vllm:d1", "fsdp:d1", marks=pytest.mark.vllm),
    ],
)
def test_vlm_grpo(tmp_path_factory, rollout_backend, actor_backend):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    )
    dataset_path = get_dataset_path(
        "/storage/openpsi/data/hiyouga__geometry3k/",
        "hiyouga/geometry3k",
    )

    example_file = "examples/vlm/geometry3k_grpo.py"
    config_name = "examples/vlm/geometry3k_grpo.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        f"rollout.backend={rollout_backend}",
        f"actor.backend={actor_backend}",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=2",
        "valid_dataset.batch_size=2",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        timeout=1800,
    )
    assert success, "CLEVR Count 70k GRPO example failed"


@pytest.mark.skip("Currently VLM dataloading is too slow. Needs to be fixed.")
@pytest.mark.gpu
def test_vlm_sft(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    )
    dataset_path = get_dataset_path(
        "/storage/openpsi/data/BUAADreamer__clevr_count_70k",
        "BUAADreamer/clevr_count_70k",
    )

    example_file = "examples/vlm/clevr_count_70k_sft.py"
    config_name = "examples/vlm/clevr_count_70k_sft.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "actor.backend=fsdp:d1",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        timeout=600,  # tokenizing the VLM dataset for SFT takes a long time
    )
    assert success, "CLEVR Count 70k SFT example failed"


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_gsm8k_ppo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_rl.py"
    config_name = "examples/math/gsm8k_ppo.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "actor.backend=fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "critic.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        f"critic.path={model_path}",
        "scheduler.type=local",
    )
    assert success, "GSM8K PPO example failed"


@pytest.mark.sglang
@pytest.mark.gpu
def test_gsm8k_ppo_colocate(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_rl.py"
    config_name = "examples/math/gsm8k_ppo.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d2",
        "actor.backend=fsdp:d2",
        "+actor.weight_update_mode=disk",
        "+rollout.scheduling_strategy.type=colocation",
        "+rollout.scheduling_strategy.target=actor",
        "critic.scheduling_strategy.type=colocation",
        "critic.scheduling_strategy.target=actor",
        "ref.scheduling_strategy.type=colocation",
        "ref.scheduling_strategy.target=actor",
        "enable_offload=True",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "sglang.mem_fraction_static=0.3",
        "vllm.gpu_memory_utilization=0.3",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "critic.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        f"critic.path={model_path}",
        "scheduler.type=local",
    )
    assert success, "GSM8K PPO colocated example failed"


@pytest.mark.ci
@pytest.mark.parametrize(
    "rollout_backend,actor_backend",
    [
        pytest.param("sglang:d1", "fsdp:d1", marks=pytest.mark.sglang),
        pytest.param("vllm:d1", "fsdp:d1", marks=pytest.mark.vllm),
    ],
)
@pytest.mark.multi_gpu
def test_gsm8k_grpo_lora(tmp_path_factory, rollout_backend, actor_backend):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/math/gsm8k_rl.py"
    config_name = "examples/math/gsm8k_grpo_lora.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        f"rollout.backend={rollout_backend}",
        f"actor.backend={actor_backend}",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        "actor.weight_update_mode=disk",
    )
    assert success, "GSM8K GRPO LoRA example failed"


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_multi_turn_math(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/multi_turn_math/gsm8k_rl_mt.py"
    config_name = "examples/multi_turn_math/gsm8k_grpo_mt.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "actor.backend=fsdp:d1",
        "gconfig.n_samples=1",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
    )
    assert success, "Multi-turn Math example failed"


@pytest.mark.gpu
def test_hhrlhf_rw(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path(
        "/storage/openpsi/data/Anthropic___hh-rlhf/", "Anthropic/hh-rlhf"
    )

    example_file = "examples/alignment/hhrlhf_rw.py"
    config_name = "examples/alignment/hhrlhf_rw.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "actor.backend=fsdp:d1",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=1",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        timeout=1800,
    )
    assert success, "HH-RLHF Reward Modeling example failed"


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_tir_grpo(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")

    example_file = "examples/tir/train_tir.py"
    config_name = "examples/tir/tir_math_config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "actor.backend=fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=64",
        "actor.mb_spec.max_tokens_per_mb=1024",
        "tir.max_length=1024",
        "train_dataset.batch_size=16",
        "valid_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
    )
    assert success, "TIR GRPO example failed"


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_search_agent_deepresearch(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    if current_platform.device_count() < 3:
        pytest.skip(
            "This test requires at least 3 GPUs (1 for LLM judge, 2 for RL) to run."
        )
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    )
    dataset_path = "/storage/openpsi/data/inclusionAI__Asearcher-train-data/ASearcher-LRM-35k.jsonl"
    if not os.path.exists(dataset_path):
        pytest.skip("Tongyi DeepResearch dataset not available")

    example_file = "examples/search_agent/tongyi_deepresearch/train.py"
    config_name = "examples/search_agent/tongyi_deepresearch/config.yaml"

    visible_devices = os.getenv(
        current_platform.device_control_env_var,
        ",".join(map(str, range(current_platform.device_count()))),
    ).split(",")
    assert len(visible_devices) >= 3

    llm_judge_exp_name = uuid.uuid4().hex
    llm_judge_trial_name = uuid.uuid4().hex

    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "judge_engine.backend=sglang:d1",
        "rollout.backend=sglang:d1",
        "actor.backend=megatron:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=128",
        "actor.mb_spec.max_tokens_per_mb=2048",
        "train_dataset.batch_size=4",
        f"train_dataset.path={dataset_path}",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "max_tokens_per_trajectory=1024",
        "max_llm_calls_per_run=2",
        f"judge_engine.experiment_name={llm_judge_exp_name}",
        f"judge_engine.trial_name={llm_judge_trial_name}",
        "scheduler.type=local",
    )
    if not success:
        raise RuntimeError("Search Agent DeepResearch example failed")


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_openai_agents(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")
    example_file = "examples/openai_agents/train_agents.py"
    config_name = "examples/openai_agents/config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "actor.backend=fsdp:d1",
        "gconfig.n_samples=1",
        "gconfig.max_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=4096",
        "train_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        "valid_dataset.batch_size=16",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
    )
    if not success:
        raise RuntimeError("OpenAI Agents example failed")


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_camel(tmp_path_factory):
    try:
        import camel.agents  # noqa
    except ImportError:
        pytest.skip("camel-ai is not installed. Skipping camel example test.")
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")
    example_file = "examples/camel/train.py"
    config_name = "examples/camel/config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "actor.backend=fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=256",
        "actor.mb_spec.max_tokens_per_mb=4096",
        "train_dataset.batch_size=16",
        f"train_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
    )
    if not success:
        raise RuntimeError("Camel Math example failed")


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_openai_proxy(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    )
    dataset_path = get_dataset_path("/storage/openpsi/data/gsm8k", "openai/gsm8k")
    example_file = "examples/agent_workflow/train.py"
    config_name = "examples/agent_workflow/config.yaml"
    success = run_async_task(
        run_example,
        example_file,
        config_name,
        "rollout.backend=sglang:d1",
        "actor.backend=fsdp:d1",
        "gconfig.n_samples=2",
        "gconfig.max_new_tokens=16",
        "gconfig.max_tokens=512",
        "actor.mb_spec.max_tokens_per_mb=4096",
        "train_dataset.batch_size=1",
        f"train_dataset.path={dataset_path}",
        "valid_dataset.batch_size=1",
        f"valid_dataset.path={dataset_path}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        "actor.weight_update_mode=xccl",
    )
    if not success:
        raise RuntimeError("OpenAI Proxy example failed")


@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_tau2(tmp_path_factory):
    """Test tau2 airline domain training with a user LLM server.

    This test requires at least 3 GPUs:
    - 1 GPU for user LLM (SGLang server simulating the user)
    - 2 GPUs for RL training (1 for rollout, 1 for training)
    """
    # Check tau2-bench is installed
    try:
        import tau2  # noqa
    except ImportError:
        pytest.skip("tau2-bench is not installed. Skipping tau2 example test.")

    # Check TAU2_DATA_DIR is set and valid
    tau2_data_dir = os.environ.get("TAU2_DATA_DIR")
    if not tau2_data_dir:
        pytest.skip("TAU2_DATA_DIR environment variable not set. Skipping tau2 test.")
    if not os.path.exists(tau2_data_dir):
        pytest.skip(
            f"TAU2_DATA_DIR ({tau2_data_dir}) does not exist. Skipping tau2 test."
        )

    # Check chat template exists
    chat_template_path = "/storage/openpsi/data/qwen3_nonthinking.jinja"
    if not os.path.exists(chat_template_path):
        pytest.skip(f"Chat template not found at {chat_template_path}")

    if current_platform.device_count() < 3:
        pytest.skip(
            "This test requires at least 3 GPUs (1 for user LLM, 2 for RL) to run."
        )

    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )

    # Get visible devices
    visible_devices = os.getenv(
        current_platform.device_control_env_var,
        ",".join(map(str, range(current_platform.device_count()))),
    ).split(",")
    assert len(visible_devices) >= 3

    # Launch user LLM server on the last GPU
    user_llm_gpu = visible_devices[-1]
    user_llm_port = 30080  # Use a different port to avoid conflicts

    _env = os.environ.copy()
    _env[current_platform.device_control_env_var] = user_llm_gpu

    logger.info(
        f"Launching user LLM server on GPU {user_llm_gpu}, port {user_llm_port}"
    )
    user_llm_proc = subprocess.Popen(
        [
            "python3",
            "-m",
            "areal.experimental.inference_service.sglang.launch_server",
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
        # Redirect to DEVNULL to avoid SGLang's progress bars cluttering the terminal
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=_env,
    )

    try:
        # Wait for user LLM server to start
        logger.info("Waiting for user LLM server to start...")
        time.sleep(60)  # SGLang server takes time to load model

        user_llm_base_url = f"http://localhost:{user_llm_port}/v1/"

        example_file = "examples/tau2/train.py"
        config_name = "examples/tau2/config_1.7b_airline.yaml"

        # Run tau2 training with first 2 GPUs
        success = run_async_task(
            run_example,
            example_file,
            config_name,
            "rollout.backend=sglang:d1",
            "actor.backend=megatron:d1",
            "gconfig.n_samples=2",
            "gconfig.max_new_tokens=1024",
            "gconfig.max_tokens=8192",
            "actor.mb_spec.max_tokens_per_mb=8192",
            "train_dataset.batch_size=2",
            "train_dataset.path=tau2/train",
            "valid_dataset.batch_size=2",
            "valid_dataset.path=tau2/test",
            "cluster.n_gpus_per_node=2",
            f"cluster.fileroot={str(experiments_path)}",
            f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
            f"actor.path={model_path}",
            "econfig.max_steps=3",  # Limit steps for faster testing
            f"econfig.user_llm_base_url={user_llm_base_url}",
            "econfig.user_llm=openai/self-hosted-qwen3",
            "actor.enable_tree_training=false",  # Disable tree training for simpler test
            "scheduler.type=local",
            "stats_logger.wandb.mode=disabled",
            timeout=600,
        )
        assert success, "Tau2 airline example failed"
    finally:
        logger.info("Shutting down user LLM server...")
        kill_process_tree(user_llm_proc.pid, graceful=False)


@pytest.mark.ci
@pytest.mark.sglang
@pytest.mark.multi_gpu
def test_openclaw_online_rl(tmp_path_factory):
    """Test openclaw online RL training via demo lifecycle (HTTP requests).

    Starts the online RL service, runs 3 episodes via HTTP (start_session →
    chat/completions → set_reward), and verifies a training step completes.
    Requires 2 GPUs: 1 for SGLang rollout, 1 for FSDP training.
    """
    # Timeout constants
    GATEWAY_STARTUP_TIMEOUT = 600  # 10 min for model loading + service init
    SETTLE_TIME = 5  # Wait for gateway to be fully ready
    HEALTH_CHECK_TIMEOUT = 10
    SESSION_NEW_TIMEOUT = 30
    SESSION_REFRESH_TIMEOUT = 130  # Longer timeout for refresh (waits for training)
    CHAT_TIMEOUT = 120
    REWARD_TIMEOUT = 10
    TRAINING_STEP_TIMEOUT = 300  # 5 min for training step

    def wait_for_pattern(process, pattern, timeout, raise_on_exit=True):
        """Wait for a regex pattern in process stdout.

        Returns the match object if found, None if timeout or process exits.
        Raises RuntimeError if process exits and raise_on_exit=True.
        """
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    if raise_on_exit:
                        raise RuntimeError(
                            f"Process terminated unexpectedly (code {process.returncode})"
                        )
                    return None
                time.sleep(0.1)
                continue
            line = line.strip()
            if line:
                logger.info(f"[Online RL] {line}")
            match = pattern.search(line)
            if match:
                return match
        return None

    if current_platform.device_count() < 2:
        pytest.skip("This test requires at least 2 GPUs to run.")

    experiments_path = tmp_path_factory.mktemp("experiments")
    name_resolve_path = tmp_path_factory.mktemp("name_resolve")
    model_path = get_model_path(
        "/storage/openpsi/models/Qwen__Qwen3-0.6B", "Qwen/Qwen3-0.6B"
    )

    admin_api_key = "test-admin-key"
    batch_size = 2

    cmd = [
        "python3",
        "examples/openclaw/train.py",
        "--config",
        "examples/openclaw/config.yaml",
        "rollout.backend=sglang:d1",
        "actor.backend=fsdp:d1",
        "gconfig.n_samples=1",
        "gconfig.max_new_tokens=128",
        "actor.mb_spec.max_tokens_per_mb=2048",
        f"train_dataset.batch_size={batch_size}",
        "cluster.n_gpus_per_node=2",
        f"cluster.fileroot={str(experiments_path)}",
        f"cluster.name_resolve.nfs_record_root={str(name_resolve_path)}",
        f"actor.path={model_path}",
        "scheduler.type=local",
        f"rollout.agent.admin_api_key={admin_api_key}",
        "stats_logger.wandb.mode=disabled",
    ]

    logger.info(f"Starting online RL service: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    gateway_pattern = re.compile(r"Proxy gateway available at (http://[0-9.:]+)")

    try:
        # Wait for gateway URL
        match = wait_for_pattern(
            process, gateway_pattern, GATEWAY_STARTUP_TIMEOUT, raise_on_exit=True
        )
        if not match:
            raise RuntimeError("Timed out waiting for gateway URL")
        gateway_url = match.group(1)
        logger.info(f"Gateway URL: {gateway_url}")

        time.sleep(SETTLE_TIME)

        import requests

        # Health check
        health_resp = requests.get(
            f"{gateway_url}/health", timeout=HEALTH_CHECK_TIMEOUT
        )
        assert health_resp.status_code == 200, (
            f"Health check failed: {health_resp.text}"
        )

        # Run batch_size + 1 episodes to trigger training
        session_api_key = None
        for episode in range(batch_size + 1):
            logger.info(f"Episode {episode + 1}/{batch_size + 1}")

            # Start/refresh session
            start_body = {"task_id": "test-task"}
            if session_api_key is not None:
                start_body["api_key"] = session_api_key
            start_resp = requests.post(
                f"{gateway_url}/rl/start_session",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {admin_api_key}",
                },
                json=start_body,
                timeout=SESSION_REFRESH_TIMEOUT
                if session_api_key
                else SESSION_NEW_TIMEOUT,
            )
            assert start_resp.status_code == 200, (
                f"start_session failed (ep {episode + 1}): {start_resp.text}"
            )
            session_data = start_resp.json()
            session_api_key = session_data["api_key"]
            logger.info(f"Session: {session_data['session_id']}")

            # Chat completion
            chat_resp = requests.post(
                f"{gateway_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {session_api_key}",
                },
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "max_tokens": 128,
                },
                timeout=CHAT_TIMEOUT,
            )
            assert chat_resp.status_code == 200, (
                f"chat/completions failed (ep {episode + 1}): {chat_resp.text}"
            )
            logger.info(f"Completion: {chat_resp.json().get('id', '')}")

            # Set reward
            reward_resp = requests.post(
                f"{gateway_url}/rl/set_reward",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {session_api_key}",
                },
                json={"reward": 1.0},
                timeout=REWARD_TIMEOUT,
            )
            assert reward_resp.status_code == 200, (
                f"set_reward failed (ep {episode + 1}): {reward_resp.text}"
            )

        # Wait for training step
        logger.info("Waiting for training step to complete...")
        match = wait_for_pattern(
            process, SUCCESS_PATTERN, TRAINING_STEP_TIMEOUT, raise_on_exit=False
        )
        assert match is not None, "Training step did not complete within timeout"
        logger.info("Training step completed!")

    finally:
        logger.info("Shutting down online RL service...")
        kill_process_tree(process.pid, graceful=False)
