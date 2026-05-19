import sys

from transformers import PreTrainedTokenizer
from datasets import load_dataset

# from areal import SFTTrainer
from areal.trainer import SFTTrainer
from areal.api.cli_args import SFTConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.data import collate_samples_to_list
from areal.utils.dataloader import create_dataloader


def get_dataset(path: str, tokenizer: PreTrainedTokenizer):
    dataset = load_dataset("json", data_files=path, split="train")
    return dataset


class Trainer(SFTTrainer):
    tokenizer: PreTrainedTokenizer = None

    def __init__(
        self, config, tokenizer: PreTrainedTokenizer, train_dataset, valid_dataset=None
    ):
        super().__init__(config, train_dataset, valid_dataset)
        self.tokenizer = tokenizer

    def _create_dataloader(self, dataset, dataset_config, rank, world_size):
        def collate_fn(samples: list[dict]):
            results = []
            for sample in samples:
                seq_token = self.tokenizer.encode(
                    sample["prompt"] + sample["answer"] + self.tokenizer.eos_token
                )
            prompt_token = self.tokenizer.encode(sample["prompt"])
            loss_mask = [0] * len(prompt_token) + [1] * (
                len(seq_token) - len(prompt_token)
            )
            results.append({"input_ids": seq_token, "loss_mask": loss_mask})
            return collate_samples_to_list(results)

        return create_dataloader(
            dataset,
            rank=rank,
            world_size=world_size,
            dataset_config=dataset_config,
            collate_fn=collate_fn,
        )


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

    with Trainer(
        config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
