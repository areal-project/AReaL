"""GRPO training on OSWorld (desktop-control) tasks with Qwen2.5-VL.

Run (from AReaL repo root, inside this docker):

    python -m examples.osworld.train --config examples/osworld/config_osworld_sglang.yaml

Before launching, make sure the OSWorld docker provider prerequisites (KVM +
``xlang/osworld-docker`` image) are satisfied on the host. See
``AReaL/examples/osworld/README.md`` for details.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
from pathlib import Path

from datasets import Dataset

from examples.osworld.osworld_config import OSWorldAgentConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils import seeding
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger

WORKFLOW_PATH = "examples.osworld.workflow.osworld_workflow.OSWorldWorkflow"


def _resolve_osworld_paths(config: OSWorldAgentConfig) -> tuple[str, str, str]:
    """Return (osworld_root, evaluation_examples_dir, test_meta_path)."""
    if config.osworld_root:
        osworld_root = os.path.abspath(config.osworld_root)
    else:
        osworld_root = str((Path(__file__).resolve().parents[3] / "OSWorld").resolve())

    evaluation_examples_dir = (
        os.path.abspath(config.evaluation_examples_dir)
        if config.evaluation_examples_dir
        else os.path.join(osworld_root, "evaluation_examples")
    )

    test_meta_path = (
        os.path.abspath(config.test_meta_path)
        if config.test_meta_path
        else os.path.join(evaluation_examples_dir, "test_small.json")
    )
    return osworld_root, evaluation_examples_dir, test_meta_path


def _build_tasks_dataset(evaluation_examples_dir: str, test_meta_path: str) -> Dataset:
    """Load OSWorld task metas into a flat Hugging Face Dataset.

    Each row mirrors the on-disk example JSON so the workflow can consume it
    without re-reading files during rollout.
    """
    with open(test_meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    rows: list[dict] = []
    for domain, example_ids in meta.items():
        for example_id in example_ids:
            example_path = (
                Path(evaluation_examples_dir)
                / "examples"
                / domain
                / f"{example_id}.json"
            )
            with open(example_path, encoding="utf-8") as f:
                task = json.load(f)
            rows.append(
                {
                    "domain": domain,
                    "example_id": example_id,
                    "id": task.get("id", example_id),
                    "instruction": task["instruction"],
                    # Keep the full task dict as a JSON string to avoid the
                    # schema explosion that datasets infers from nested dicts.
                    "task_config_json": json.dumps(task, ensure_ascii=False),
                }
            )
    if not rows:
        raise ValueError(
            f"No OSWorld tasks found under {evaluation_examples_dir} "
            f"via meta file {test_meta_path}"
        )
    return Dataset.from_list(rows)


def main(args):
    config, _ = load_expr_config(args, OSWorldAgentConfig)

    rank = int(os.getenv("RANK", "0"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")

    osworld_root, evaluation_examples_dir, test_meta_path = _resolve_osworld_paths(
        config
    )

    dataset = _build_tasks_dataset(evaluation_examples_dir, test_meta_path)

    # The workflow only needs the task config dict; inflate on the fly.
    def _inflate(row):
        row.update(json.loads(row.pop("task_config_json")))
        return row

    dataset = dataset.map(_inflate)

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        evaluation_examples_dir=evaluation_examples_dir,
        osworld_root=osworld_root,
        provider_name=config.provider_name,
        path_to_vm=config.path_to_vm,
        os_type=config.os_type,
        headless=config.headless,
        screen_size=(config.screen_width, config.screen_height),
        observation_type=config.observation_type,
        action_space=config.action_space,
        cache_dir=config.osworld_cache_dir,
        max_steps=config.max_steps,
        n_trajs=config.n_trajs,
        sleep_after_execution=config.sleep_after_execution,
        env_reset_wait_secs=config.env_reset_wait_secs,
        max_workers=config.max_workers,
        turn_discount=config.turn_discount,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        remote_server_url=config.remote_server_url,
        remote_request_timeout_secs=config.remote_request_timeout_secs,
        gateway_endpoint=config.gateway_endpoint,
        gateway_token=config.gateway_token,
        gateway_timeout_secs=config.gateway_timeout_secs,
        text_only=config.text_only,
        # Default the VL processor to the actor checkpoint dir; AutoProcessor
        # picks up the matching preprocessor_config.json there. Workflow only
        # consumes this when text_only=False.
        processor_path=config.actor.path,
    )

    eval_workflow_kwargs = workflow_kwargs.copy()

    with PPOTrainer(
        config,
        train_dataset=dataset,
        valid_dataset=dataset,
    ) as trainer:
        trainer.train(
            workflow=WORKFLOW_PATH,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=WORKFLOW_PATH,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
