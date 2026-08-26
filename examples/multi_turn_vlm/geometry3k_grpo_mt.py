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

    # Reward / grading / feedback live in the env (calc_score tool); the agent
    # is task-agnostic. env_factory is a dotted import path resolved in the
    # rollout worker; env_args must be JSON-serializable.
    #
    # Token bookkeeping, vision tensors, reward propagation (turn_discount) and
    # trajectory export shape (export_style) are handled by the OpenAI proxy and
    # configured under `rollout.agent` in the YAML.
    workflow_kwargs = dict(
        env_factory="examples.multi_turn_vlm.geo3k_env.Geo3kCalcScoreEnv",
        env_args={"max_turns": config.max_turns, "tool_format": config.tool_format},
        max_turns=config.max_turns,
        max_completion_tokens=config.gconfig.max_new_tokens,
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        # Cap a trajectory to one microbatch so multi-turn sequences never
        # exceed the FFD packing capacity (they cannot be split for VLM).
        max_tokens_per_traj=config.actor.mb_spec.max_tokens_per_mb,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    agent = "areal.workflow.vision_multiturn.VisionMultiTurnAgent"
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow=agent,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=agent,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
