# SPDX-License-Identifier: Apache-2.0
"""Launch the standard Arena workflow with Qwen runtime options."""

import json
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path


def decorate_controller(controller):
    initialize = controller.initialize

    def initialize_model(**kwargs):
        server_args = dict(kwargs["server_args"])
        server_args.update(
            linear_attn_prefill_backend="flashinfer",
            linear_attn_decode_backend="flashinfer",
            ple_offload_embedding=True,
        )
        return initialize(**{**kwargs, "server_args": server_args})

    controller.initialize = initialize_model
    start_proxy = controller.start_proxy

    def start():
        scheduler = controller.scheduler
        fork_workers = scheduler.fork_workers

        def fork(*args, **kwargs):
            if (
                kwargs.get("command")
                == "areal.experimental.openai.proxy.proxy_rollout_server"
            ):
                kwargs["command"] = "examples.swe.qwen38_flash_next.proxy"
            return fork_workers(*args, **kwargs)

        scheduler.fork_workers = fork
        try:
            return start_proxy()
        finally:
            scheduler.fork_workers = fork_workers

    controller.start_proxy = start
    return controller


def select_task_indices(data_ids, selected):
    """Select an explicit single-stream subset, preserving the requested order."""
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(x, str) or not x for x in selected)
    ):
        raise ValueError("Task selection must be a nonempty JSON list of data IDs")
    if len(set(selected)) != len(selected):
        raise ValueError("Task selection contains duplicate data IDs")
    if len(set(data_ids)) != len(data_ids):
        raise ValueError("Task selection requires unique data IDs in one stream")
    positions = {key: index for index, key in enumerate(data_ids)}
    if any(key not in positions for key in selected):
        raise ValueError(
            "Task selection contains IDs absent from the configured stream"
        )
    return [positions[key] for key in selected]


def select_evaluation_rows(rows: list[dict], selected: list[str]) -> list[dict]:
    """Pin explicit versions while retaining stream routing for the same env keys."""
    select_task_indices(selected, selected)

    def env_key(data_id: str) -> str:
        if not isinstance(data_id, str) or not data_id.startswith("env:"):
            raise ValueError(
                "Evaluation task IDs must be explicit env:key@version refs"
            )
        parts = data_id[4:].split("@")
        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "Evaluation task IDs must be explicit env:key@version refs"
            )
        return parts[0]

    source = {}
    for row in rows:
        key = env_key(row["data_id"])
        if key in source:
            raise ValueError("Evaluation source must contain unique environment keys")
        source[key] = row
    selected_keys = [env_key(data_id) for data_id in selected]
    if len(set(selected_keys)) != len(selected_keys):
        raise ValueError("Evaluation selection must contain unique environment keys")
    if any(key not in source for key in selected_keys):
        raise ValueError(
            "Evaluation selection contains environments absent from stream"
        )
    return [
        {**source[key], "data_id": data_id}
        for key, data_id in zip(selected_keys, selected, strict=True)
    ]


def configure_training_rpc(scheduler):
    """Allow cold 256K steps without replaying a timed-out optimizer update."""
    original = scheduler.async_call_engine

    async def call_engine(worker_id, method, engine_name=None, *args, **kwargs):
        if method == "ppo_update":
            kwargs["http_timeout"] = 8 * 3600
            kwargs["max_retries"] = 1
        return await original(worker_id, method, engine_name, *args, **kwargs)

    scheduler.async_call_engine = call_engine
    return scheduler


def select_arena_dataset(dataset, selected: list[str]):
    """Use explicit Env versions for both training and evaluation selections."""
    from datasets import Dataset

    return Dataset.from_list(select_evaluation_rows(list(dataset), selected))


def validate_evaluation_only(config, dataset):
    """Reject settings that can train, recover weights, or omit evaluation tasks."""
    if config.total_train_steps != 0 or not config.evaluator.eval_before_train:
        raise ValueError("swe-eval requires zero training steps and eval_before_train")
    if config.recover.mode not in ("off", "disabled"):
        raise ValueError("swe-eval requires recovery to be disabled")
    if config.eval_gconfig.n_samples != 1:
        raise ValueError("swe-eval requires eval_gconfig.n_samples=1")
    valid = config.valid_dataset
    if valid is None or valid.shuffle or valid.drop_last:
        raise ValueError(
            "swe-eval requires an unshuffled validation set without drop_last"
        )
    if not len(dataset) or valid.batch_size != len(dataset):
        raise ValueError(
            "swe-eval validation batch size must equal the selected task count"
        )


def configure_evaluation_only(config, task_count):
    """Reuse the MM recipe for one complete, fixed-weight benchmark pass."""
    from areal.api.cli_args import ValidDatasetConfig

    if task_count < 1:
        raise ValueError("swe-eval requires a nonempty task selection")
    config.total_train_epochs = 1
    config.total_train_steps = 0
    config.recover.mode = "disabled"
    config.evaluator.eval_before_train = True
    config.gconfig.n_samples = config.eval_gconfig.n_samples = 1
    config.gconfig.reward_normalization = False
    config.gconfig.max_new_tokens = config.eval_gconfig.max_new_tokens = 32768
    config.rollout.max_concurrent_rollouts = 64
    config.rollout.queue_size = config.rollout.consumer_batch_size = task_count
    config.rollout.max_head_offpolicyness = 0
    config.actor.min_usable_group_size = 1
    config.actor.mb_spec.n_mbs = ((task_count + 7) // 8) * 8
    config.actor.mb_spec.n_mbs_divisor = 8
    config.actor.optimizer.lr = 0.0
    config.train_dataset.batch_size = task_count
    config.valid_dataset = ValidDatasetConfig(
        **{**asdict(config.train_dataset), "shuffle": False, "drop_last": False}
    )


def main(profile, args):
    if profile not in ("swe", "swe-eval"):
        raise ValueError("Expected swe or swe-eval")
    evaluation_only = profile == "swe-eval"
    from examples.swe.train_swe_rl import get_arena_mixture_dataset
    from examples.swe.utils import SWEPPOConfig

    from areal import PPOTrainer
    from areal.api.cli_args import load_expr_config
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.infra.scheduler.slurm import SlurmScheduler

    config, _ = load_expr_config(args, SWEPPOConfig)
    selection_file = os.environ.get("QWEN_ARENA_TASK_IDS_FILE")
    selected = json.loads(Path(selection_file).read_text()) if selection_file else None
    if evaluation_only:
        if selected is None:
            raise ValueError("swe-eval requires QWEN_ARENA_TASK_IDS_FILE")
        select_task_indices(selected, selected)
        configure_evaluation_only(config, len(selected))
    else:
        # Training pauses can outlast the harness default. Evaluation keeps its
        # stream/server defaults; explicit task env overrides win in both modes.
        config.econfig.arena_task_envs.setdefault(
            "DSH_LLM_REQUEST_TIMEOUT_SECONDS", "7200"
        )
    from examples.swe.qwen38_flash_next.batch_snapshot import (
        resolve_replay_paths,
        validate_diagnostic_replay,
    )

    replay_paths = resolve_replay_paths(
        os.environ.get("QWEN_BATCH_REPLAY_PATH"),
        os.environ.get("QWEN_BATCH_REPLAY_PATHS"),
    )
    if replay_paths:
        if evaluation_only:
            raise ValueError("Batch replay is not an evaluation mode")
        validate_diagnostic_replay(config, len(replay_paths))
    dataset, streams = get_arena_mixture_dataset(
        config.econfig, size_multiple=config.train_dataset.batch_size
    )
    if selection_file:
        if len(streams) != 1:
            raise ValueError("Task selection requires exactly one Arena stream")
        dataset = select_arena_dataset(dataset, selected)
        if len(dataset) < config.train_dataset.batch_size:
            raise ValueError("Task selection must contain at least one training batch")
    config.econfig.arena_streams = streams
    config.econfig.arena_streams_file = ""
    config.econfig.arena_streams_yaml_b64 = ""
    if evaluation_only:
        validate_evaluation_only(config, dataset)
    generation = config.eval_gconfig if evaluation_only else config.gconfig
    workflow = "examples.swe.arena_agent.ArenaStreamAgentWorkflow"
    kwargs = dict(
        econfig=asdict(config.econfig),
        gen_args=dict(
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            max_completion_tokens=generation.max_new_tokens,
        ),
        timeout=config.econfig.timeout,
    )

    class RecipeTrainer(PPOTrainer):
        def _init_scheduler(self):
            return configure_training_rpc(
                SlurmScheduler(
                    exp_config=self.config,
                    container_mounts=os.environ["QWEN_MOUNTS"],
                    startup_timeout=86400,
                )
            )

    # This public controller hook supplies SGLang options missing from the config
    # schema. Restore the factory even if initialization or training fails.
    factory = RemoteSGLangEngine.as_controller
    RemoteSGLangEngine.as_controller = staticmethod(
        lambda *args, **kwargs: decorate_controller(factory(*args, **kwargs))
    )
    try:
        with RecipeTrainer(
            config,
            train_dataset=dataset,
            valid_dataset=dataset if evaluation_only else None,
        ) as trainer:
            snapshot_dir = os.environ.get("QWEN_BATCH_SNAPSHOT_DIR")
            capture = nullcontext()
            if snapshot_dir and not evaluation_only:
                from examples.swe.qwen38_flash_next.batch_snapshot import (
                    capture_training_batches,
                )

                capture = capture_training_batches(
                    trainer.actor,
                    Path(snapshot_dir),
                    {
                        "experiment": config.experiment_name,
                        "trial": config.trial_name,
                        "model_path": config.tokenizer_path,
                        "allocation_mode": config.allocation_mode,
                        "n_samples": config.gconfig.n_samples,
                    },
                )
            replay = nullcontext()
            if replay_paths:
                from examples.swe.qwen38_flash_next.batch_snapshot import (
                    replay_training_batches,
                )

                replay = replay_training_batches(
                    trainer.actor,
                    replay_paths,
                    {
                        "model_path": config.tokenizer_path,
                        "n_samples": config.gconfig.n_samples,
                    },
                )
            # Capture wraps replay, so its artifacts describe the supplied batch.
            with replay, capture:
                trainer.train(
                    workflow=workflow,
                    workflow_kwargs=kwargs,
                    eval_workflow=workflow if evaluation_only else None,
                    eval_workflow_kwargs=kwargs if evaluation_only else None,
                )
    finally:
        RemoteSGLangEngine.as_controller = staticmethod(factory)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
