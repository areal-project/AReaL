import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path

from agent_rl_config import AgentRLConfig
from datasets import load_dataset

from areal import PPOTrainer
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.utils import seeding
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger

WORKFLOW_PATH = "workflow.camel_rlvr_workflow.CamelRLVRWorkflow"


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    rank = int(os.getenv("RANK", "0"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    assert allocation_mode.train is not None

    dataset = load_dataset(
        path="parquet",
        split="train",
        data_files=[
            str(
                Path(__file__).parent.parent.parent
                / "dataset"
                / config.train_dataset.path
            )
        ],
    )

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        n_trajs=config.n_trajs,
        max_tokens=config.max_tokens_per_trajectory,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        max_iteration=config.max_iteration,
        max_workers=config.max_workers,
        non_think_mode=config.non_think_mode,
        task_timeouts=config.task_timeouts,
        filter_uniform_reward=config.filter_uniform_reward,
        encourage_completion_reward=config.encourage_completion_reward,
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
