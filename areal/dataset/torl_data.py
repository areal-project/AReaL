# SPDX-License-Identifier: Apache-2.0

import os
import time

import requests
from datasets import load_dataset

"""
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   data_source   1275 non-null   object
 1   prompt        1275 non-null   object
 2   ability       1275 non-null   object
 3   reward_model  1275 non-null   object
 4   extra_info    1275 non-null   object
"""


TORL_DATA_URLS = [
    (
        "https://github.com/GAIR-NLP/ToRL/raw/main/data/torl_data/test.parquet",
        "/tmp/areal/torl_data/test.parquet",
    ),
    (
        "https://github.com/GAIR-NLP/ToRL/raw/main/data/torl_data/train.parquet",
        "/tmp/areal/torl_data/train.parquet",
    ),
]


def download(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {url} to {save_path}")


def prepare_torl_data(rank: int):
    if rank == 0 and (not os.path.exists("/tmp/areal/torl_data/_SUCCESS")):
        os.makedirs("/tmp/areal/torl_data", exist_ok=True)
        for url, save_path in TORL_DATA_URLS:
            download(url, save_path)
        # add SUCCESS flag file
        with open("/tmp/areal/torl_data/_SUCCESS", "w") as f:
            f.write("SUCCESS")

    TIMEOUT = 120
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        if os.path.exists("/tmp/areal/torl_data/_SUCCESS"):
            break
        time.sleep(1)
    if not os.path.exists("/tmp/areal/torl_data/_SUCCESS"):
        raise TimeoutError("Prepare ToRL data timeout")


def get_torl_data_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    raise NotImplementedError("ToRL dataset not supported in SFT training")


def get_torl_data_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    prepare_torl_data(int(os.getenv("RANK", "0")))
    # Load parquet dataset instead of json
    dataset = load_dataset("parquet", data_files=path, split="train")

    def process(sample):
        # Handle the prompt content - it might be a list of messages or a string
        answer = sample["reward_model"]["ground_truth"]
        answer = f"\\boxed{{{answer}}}"
        return {"messages": sample["prompt"], "answer": answer}

    dataset = dataset.map(process).remove_columns(["prompt", "reward_model"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
