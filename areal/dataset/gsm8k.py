# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset


def _common_prefix_len(left: list[int], right: list[int]) -> int:
    for i, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return i
    return min(len(left), len(right))


def get_gsm8k_sft_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        seq_token = tokenizer.encode(
            sample["question"] + sample["answer"] + tokenizer.eos_token
        )
        prompt_token = tokenizer.encode(sample["question"])
        # `prompt_token` is not always a token-level prefix of `seq_token`:
        # byte-level/BPE tokenizers may merge a token across the question/answer
        # join (e.g. when the question has no trailing whitespace). Use the length
        # of the common token prefix as the boundary so any boundary-spanning
        # token is attributed to the answer and supervised.
        prompt_len = _common_prefix_len(prompt_token, seq_token)
        loss_mask = [0] * prompt_len + [1] * (len(seq_token) - prompt_len)
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset


def get_gsm8k_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, name="main", split=split)

    def process(sample):
        messages = [
            {
                "role": "user",
                "content": sample["question"]
                + "\nPlease put your final answer within \\boxed{}.",
            }
        ]
        return {"messages": messages}

    dataset = dataset.map(process).remove_columns(["question"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
