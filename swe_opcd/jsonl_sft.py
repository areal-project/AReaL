import sys

from transformers import PreTrainedTokenizer
from datasets import load_dataset

# from areal import SFTTrainer
from areal.trainer import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer


def get_dataset(path: str, tokenizer: PreTrainedTokenizer):
    dataset = load_dataset("json", data_files=path, split="train")

    def process(sample: dict):
        seq_token = tokenizer.encode(
            sample["prompt"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["prompt"])
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    return dataset


def main(args):
    config, _ = load_expr_config(args, SFTConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # train_dataset = get_custom_dataset(
    #     split="train",
    #     dataset_config=config.train_dataset,
    #     tokenizer=tokenizer,
    # )
    # valid_dataset = get_custom_dataset(
    #     split="test",
    #     dataset_config=config.valid_dataset,
    #     tokenizer=tokenizer,
    # )
    train_dataset = get_dataset(path=config.train_dataset.path, tokenizer=tokenizer)
    valid_dataset = get_dataset(path=config.valid_dataset.path, tokenizer=tokenizer)

    with SFTTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
