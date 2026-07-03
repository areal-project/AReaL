import pathlib
import sys

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from config import MultiTurnVLMGRPOConfig  # noqa: E402


def main(args):
    config, _ = load_expr_config(args, MultiTurnVLMGRPOConfig)
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )
    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
        processor=processor,
    )

    # Reward / grading / feedback live in the env (calc_score tool); the workflow
    # is task-agnostic. env_factory is a dotted import path resolved in the
    # rollout worker; env_args must be JSON-serializable.
    workflow_kwargs = dict(
        env_factory="examples.multi_turn_vlm.geo3k_env.Geo3kCalcScoreEnv",
        env_args={"max_turns": config.max_turns, "tool_format": config.tool_format},
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        processor=config.tokenizer_path,
        max_turns=config.max_turns,
        turn_discount=config.turn_discount,
        # Cap a trajectory to one microbatch so multi-turn sequences never
        # exceed the FFD packing capacity (they cannot be split for VLM).
        max_tokens_per_traj=config.actor.mb_spec.max_tokens_per_mb,
        export_style=config.export_style,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.vision_multiturn.VisionMultiTurnWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.vision_multiturn.VisionMultiTurnWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
