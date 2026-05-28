import re
import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_processor_and_tokenizer


def extract_answer(pred_str, data_name, use_last_number=True):
    match = re.findall(r"\[([0-9\.]+)\]", pred_str)
    if match:
        return match[-1]

    return ""


def clevr_count_70k_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    sol = extract_answer(completions, data_name="")  # str number
    ans = answer

    if sol is None:
        return 0
    if ans is None:
        return 0

    return float(sol.strip() == ans.strip())


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
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

    workflow_kwargs = dict(
        reward_fn="examples.vlm.clevr_count_70k_grpo.clevr_count_70k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        processor=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.vision_rlvr.VisionRLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
