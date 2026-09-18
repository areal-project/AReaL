# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from datasets import Dataset, load_dataset

from areal.utils.hf_utils import apply_chat_template


def get_dapo_math_17k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
) -> Dataset:
    """Load local DAPO-Math-17k parquet data for RL training."""
    del split
    dataset_root = Path(path)
    parquet_files = sorted((dataset_root / "data").glob("*.parquet"))
    json_files = sorted(dataset_root.glob("*.jsonl")) + sorted(
        dataset_root.glob("*.json")
    )
    if parquet_files:
        dataset = load_dataset(
            "parquet",
            data_files=[str(data_file) for data_file in parquet_files],
            split="train",
        )
    elif json_files:
        dataset = load_dataset(
            "json",
            data_files=[str(data_file) for data_file in json_files],
            split="train",
        )
    else:
        raise FileNotFoundError(
            f"No parquet or JSON data files found under {dataset_root}"
        )

    def process(sample):
        answer = sample.get("label")
        if answer is None:
            answer = sample["reward_model"]["ground_truth"]
        return {"messages": sample["prompt"], "answer": str(answer)}

    dataset = dataset.map(process, remove_columns=dataset.column_names)

    if max_length is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when max_length is set")

        def filter_length(sample):
            prompt_ids = apply_chat_template(
                tokenizer, sample["messages"], add_generation_prompt=True, tokenize=True
            )
            return len(prompt_ids) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
