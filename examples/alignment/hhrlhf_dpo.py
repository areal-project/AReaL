import sys

from areal import DPOTrainer
from areal.api.cli_args import DPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def main(args):
    config, _ = load_expr_config(args, DPOConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split=config.train_dataset.split,
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )
    valid_dataset = get_custom_dataset(
        split=config.valid_dataset.split if config.valid_dataset is not None else None,
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    with DPOTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
