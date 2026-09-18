# SPDX-License-Identifier: Apache-2.0
"""Launch the standard SWE/GSM8K workflow with Qwen runtime options."""

import json
import os
import sys
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


def main(profile, args):
    from examples.swe.train_swe_rl import get_arena_mixture_dataset
    from examples.swe.utils import SWEPPOConfig

    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.infra.scheduler.slurm import SlurmScheduler
    from areal.utils.hf_utils import load_hf_tokenizer

    config, _ = load_expr_config(args, SWEPPOConfig if profile == "swe" else GRPOConfig)
    if profile == "swe":
        dataset, streams = get_arena_mixture_dataset(
            config.econfig, size_multiple=config.train_dataset.batch_size
        )
        selection_file = os.environ.get("QWEN_ARENA_TASK_IDS_FILE")
        if selection_file:
            if len(streams) != 1:
                raise ValueError("Task selection requires exactly one Arena stream")
            selected = json.loads(Path(selection_file).read_text())
            dataset = dataset.select(
                select_task_indices(list(dataset["data_id"]), selected)
            )
            if len(dataset) < config.train_dataset.batch_size:
                raise ValueError(
                    "Task selection must contain at least one training batch"
                )
        config.econfig.arena_streams = streams
        config.econfig.arena_streams_file = ""
        config.econfig.arena_streams_yaml_b64 = ""
        workflow = "examples.swe.arena_agent.ArenaStreamAgentWorkflow"
        kwargs = dict(
            econfig=asdict(config.econfig),
            gen_args=dict(
                temperature=config.gconfig.temperature,
                top_p=config.gconfig.top_p,
                top_k=config.gconfig.top_k,
                max_completion_tokens=config.gconfig.max_new_tokens,
            ),
            timeout=config.econfig.timeout,
        )
    else:
        dataset = get_custom_dataset(
            split="train",
            dataset_config=config.train_dataset,
            tokenizer=load_hf_tokenizer(config.tokenizer_path),
        )
        workflow = "areal.workflow.openai.math_agent.MathAgent"
        kwargs = dict(
            temperature=config.gconfig.temperature,
            top_p=config.gconfig.top_p,
            max_completion_tokens=config.gconfig.max_new_tokens,
        )

    class RecipeTrainer(PPOTrainer):
        def _init_scheduler(self):
            return SlurmScheduler(
                exp_config=self.config, container_mounts=os.environ["QWEN_MOUNTS"]
            )

    # This public controller hook supplies SGLang options missing from the config
    # schema. Restore the factory even if initialization or training fails.
    factory = RemoteSGLangEngine.as_controller
    RemoteSGLangEngine.as_controller = staticmethod(
        lambda *args, **kwargs: decorate_controller(factory(*args, **kwargs))
    )
    try:
        with RecipeTrainer(
            config, train_dataset=dataset, valid_dataset=None
        ) as trainer:
            trainer.train(workflow=workflow, workflow_kwargs=kwargs, eval_workflow=None)
    finally:
        RemoteSGLangEngine.as_controller = staticmethod(factory)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
