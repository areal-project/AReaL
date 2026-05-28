import sys

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.reward import get_math_verify_worker
from areal.utils import logging
from areal.utils.hf_utils import load_hf_tokenizer

from tir_workflow import TIRGRPOConfig  # isort: skip

logger = logging.getLogger("TIR Training")


def math_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    # tool_using = 0.01 if 'tool_using' in kwargs and kwargs['tool_using'] else 0
    # tool_success = 0.05 if 'tool_status' in kwargs and kwargs['tool_status'] else 0

    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        return 0.0


def main(args):
    config, _ = load_expr_config(args, TIRGRPOConfig)

    logger.info("Starting TIR training")
    logger.info(f"Configuration: {config.experiment_name}")
    logger.info(f"Model: {config.actor.path}")
    logger.info(f"Batch size: {config.train_dataset.batch_size}")

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load datasets
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )

    workflow_kwargs = dict(
        reward_fn="examples.tir.train_tir.math_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        tir_config=config.tir,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    # Create trainer
    with PPOTrainer(config, train_dataset, valid_dataset) as trainer:
        # Run training
        trainer.train(
            workflow="examples.tir.tir_workflow.TIRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.tir.tir_workflow.TIRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
