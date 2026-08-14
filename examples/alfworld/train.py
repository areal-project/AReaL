"""ALFWorld GRPO Training with AReaL.

Usage:
    # Step 1: Build dataset
    python examples/alfworld/dataset.py \
        --data_root /storage/openpsi/users/yl/agent-memory/MemRL/data/alfworld/json_2.1.1 \
        --output_dir /tmp/areal/alfworld_dataset \
        --split train

    # Step 2: Train
    python examples/alfworld/train.py \
        --config examples/alfworld/config.yaml \
        scheduler.type=local
"""
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))
from config import ALFWorldConfig

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, ALFWorldConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    max_steps = config.agent_run_args.get("max_steps", 30)

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        export_style=config.export_style,
        max_steps=max_steps,
        turn_discount=config.turn_discount,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.alfworld.workflow.ALFWorldWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.alfworld.workflow.ALFWorldWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
