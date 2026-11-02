"""Qwen3.5 PointerHead decision SFT entrypoint.

Single-NPU PoC:

    MODEL_PATH=/path/to/Qwen3.5-4B-Base \
    SUITE_PATH=/path/to/decision-suite \
      bash examples/decision/run_npu_poc.sh

Or, from the repo root:

    python3 examples/decision/train_decision.py \
        --config examples/decision/qwen3_5_decision_sft_npu_poc.yaml \
        actor.path=/path/to/Qwen3.5-4B-Base \
        train_dataset.path=/path/to/train.jsonl \
        valid_dataset.path=/path/to/development.jsonl
"""

import sys

from areal import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, SFTConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = None
    if config.valid_dataset is not None:
        valid_dataset = get_custom_dataset(
            dataset_config=config.valid_dataset,
            tokenizer=tokenizer,
        )

    with SFTTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
