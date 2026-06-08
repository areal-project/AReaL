# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset


def get_hhrlhf_rw_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        chosen_seq_token = tokenizer.encode(sample["chosen"] + tokenizer.eos_token)
        rejected_seq_token = tokenizer.encode(sample["rejected"] + tokenizer.eos_token)
        return {"chosen_ids": chosen_seq_token, "rejected_ids": rejected_seq_token}

    dataset = dataset.map(process).remove_columns(["chosen", "rejected"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(
            lambda x: (
                (len(x["chosen_ids"]) <= max_length)
                and (len(x["rejected_ids"]) <= max_length)
            )
        )

    return dataset


def get_hhrlhf_dpo_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
):
    """Load HH-RLHF dataset for DPO training.

    Each sample will contain:
    - ``chosen_ids`` / ``rejected_ids``: full token ids (prompt + response).
    - ``chosen_loss_mask`` / ``rejected_loss_mask``: boolean mask where ``True``
      marks the response tokens that participate in the loss.

    Reference log-probabilities are computed online by the ref engine during
    training (configured via the ``ref`` field in ``DPOConfig``).
    """
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        chosen_ids = tokenizer.encode(sample["chosen"] + tokenizer.eos_token)
        rejected_ids = tokenizer.encode(sample["rejected"] + tokenizer.eos_token)

        prompt_len = 0
        for c, r in zip(chosen_ids, rejected_ids):
            if c == r:
                prompt_len += 1
            else:
                break

        return {
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "chosen_loss_mask": [0] * prompt_len + [1] * (len(chosen_ids) - prompt_len),
            "rejected_loss_mask": [0] * prompt_len
            + [1] * (len(rejected_ids) - prompt_len),
        }

    dataset = dataset.map(process).remove_columns(["chosen", "rejected"])

    if max_length is not None:
        dataset = dataset.filter(
            lambda x: (len(x["chosen_ids"]) <= max_length)
            and (len(x["rejected_ids"]) <= max_length)
        )

    return dataset
