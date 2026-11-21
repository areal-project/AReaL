# Dataset

**AReaL** directly integrates with the `Dataset` class from the HuggingFace `datasets`
package. This gives you full flexibility to load, process, and filter your data before
training.

The required column names (keys) and data format depend on the specific implementation
of the agent workflow (for online reinforcement learning) or the training engines (for
offline training, such as `LMEngine` for Supervised Fine-Tuning (SFT)).

Here are two concrete examples from the existing implementation:

## SFT (Offline Training)

In the SFT example, we see that the loaded data is directly passed to the `train_lm`
method:

```python
# areal/trainer/sft_trainer.py
for global_step in range(start_step, max_steps):
    batch = self._load_bcast_from(data_generator)
    self.actor.train_lm(batch)
```

In this case, the `train_lm` method requires the keys "input_ids", "attention_mask", and
"loss_mask" to function. We first tokenize the dataset to extract the "input_ids" and
"loss_mask". Then, the `pad_sequences_to_tensors` method is used to batch multiple
sequences and append the "attention_mask":

```python
# areal/dataset/gsm8k.py
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
        loss_mask = [0] * len(prompt_token) + [1] * (len(seq_token) - len(prompt_token))
        return {"input_ids": seq_token, "loss_mask": loss_mask}

    dataset = dataset.map(process).remove_columns(["question", "answer"])

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    return dataset
```

## GRPO (Online Training)

In the GRPO example, the loaded data is first used for inference rather than training:

```python
# areal/trainer/rl_trainer.py
self.train_dataloader = self._create_dataloader(
    train_dataset,
    dataset_config=self.config.train_dataset,
    rank=self.actor.data_parallel_rank,
    world_size=self.actor.data_parallel_world_size,
)
for global_step in range(start_step, max_steps):
    rollout_batch = self.actor.prepare_batch(
        self.train_dataloader,
        workflow=workflow,
        workflow_kwargs=workflow_kwargs,
        should_accept_fn=dynamic_filter_fn,
        group_size=config.gconfig.n_samples,
        dynamic_bs=self.config.dynamic_bs,
    )
```

Note that the `collate_fn` here is an identity function, meaning it simply returns the
list of individual data items as a batch. In `prepare_batch`, the data is then
dispatched to multiple concurrent executions of workflows, where each dispatched data
corresponds to a single episode.

In the following sections, we take
[`RLVRWorkflow`](https://github.com/areal-project/AReaL/blob/main/areal/workflow/rlvr.py)
as an example. Agent workflows have the same pattern of using input data. You are free
to modify the customized dataset to include any keys as long as they accord with your
workflow implementation.

The `RLVRWorkflow` implementation extracts the "messages" field from the data dictionary
as the prompt for generating a response. Additionally, this data is passed to the
`reward_fn` as keyword arguments, which allows the reward function to make use of other
dataset fields, like "answers". Here’s an example:

```python
# areal/workflow/rlvr.py
class RLVRWorkflow(RolloutWorkflow):

    async def arun_episode(self, engine: InferenceEngine, data):
        input_ids = self.tokenizer.apply_chat_template(
            data["messages"],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        req = ModelRequest(
            input_ids=input_ids,
            ...
        )
        ...
        reward = self.reward_fn(
            prompt=prompt_str,
            completions=completions_str,
            prompt_ids=resp.input_tokens,
            completion_ids=resp.output_tokens,
            **data,
        )
```

Thus, the "messages" field must be constructed when loading the dataset, and the reward
function should be defined to handle the dataset's specific fields. Here’s how you can
process the dataset for this example:

```python
from datasets import load_dataset

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
```

## Offline prepared records

The generic loader branch accepts records that you have already prepared and saved with
`Dataset.save_to_disk()`. It reads those records from local disk and does not tokenize
them, apply a chat template, or filter them by `max_length`. For text RL, `RLVRWorkflow`
uses `messages` to build the generation prompt and passes every record field to the
reward function. In the GSM8K example, `answer` therefore reaches `gsm8k_reward_fn`; the
validator below only checks that `answer` exists and does not grade the answer.

The validator has two local profiles. `text-rl` checks object-shaped records, non-empty
`messages`, supported roles, and string message content. `tokenized-sft` checks
non-empty `input_ids` and `loss_mask`, matching lengths, non-negative non-boolean token
IDs, masks that are booleans or exact numeric `0`/`1`, and at least one supervised
position. It can also enforce `--max-length`.

The JSON report contains `profile`, `checked`, `valid`, `invalid`,
`diagnostics_truncated`, and ordered `diagnostics`. Exit code 0 means the input is
non-empty and valid; 1 means the selected input is empty or has invalid records; 2 means
an argument, path, load, split, or report error. The `--max-errors` option limits stored
diagnostics, not the number of records scanned. A persisted `DatasetDict` with `train`
and `validation` requires `--split train` or `--split validation` to select one
partition. JSONL is read one physical line at a time, so memory is affected by the
largest line, runtime libraries, and stored diagnostics rather than the whole file.
Hugging Face directories use the local `datasets` Arrow loader, whose memory behavior
belongs to that library.

Adapter matching uses the complete path, including parent directories. Keep the
prepared-data directory and its parents away from names such as `gsm8k` and
`geometry3k`. The temporary-directory prefix below avoids those names.

Run the following complete example from the AReaL repository root. It creates three
local math records, saves a `Dataset` and a `DatasetDict`, reloads both through the
existing loader, checks the selected split, and calls the real validator module before
the temporary directory is removed.

<!-- areal: offline-dataset-example start -->

```python
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset, DatasetDict

from areal.dataset import get_custom_dataset


records = [
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 1 + 1? Put the final answer within \\boxed{}.",
            }
        ],
        "answer": "2",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 2 + 3? Put the final answer within \\boxed{}.",
            }
        ],
        "answer": "5",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 6 - 4? Put the final answer within \\boxed{}.",
            }
        ],
        "answer": "2",
    },
]

with TemporaryDirectory(prefix="areal-offline-math-") as work_dir:
    work = Path(work_dir)
    dataset_dir = work / "offline_math_dataset"
    split_dir = work / "offline_math_splits"

    Dataset.from_list(records).save_to_disk(dataset_dir)
    loaded_records = list(
        get_custom_dataset(path=str(dataset_dir), type="rl", split=None)
    )
    assert len(loaded_records) == 3
    for original, loaded_record in zip(records, loaded_records, strict=True):
        assert loaded_record["messages"] == original["messages"]
        assert loaded_record["answer"] == original["answer"]

    DatasetDict(
        {
            "train": Dataset.from_list(records[:2]),
            "validation": Dataset.from_list(records[2:]),
        }
    ).save_to_disk(split_dir)
    loaded_train = list(
        get_custom_dataset(path=str(split_dir), type="rl", split="train")
    )
    loaded_validation = list(
        get_custom_dataset(path=str(split_dir), type="rl", split="validation")
    )
    assert len(loaded_train) == 2
    assert len(loaded_validation) == 1

    try:
        get_custom_dataset(path=str(split_dir), type="rl", split="test")
    except ValueError as error:
        assert "test" in str(error)
        assert "train" in str(error)
        assert "validation" in str(error)
    else:
        raise AssertionError("Expected a missing-split ValueError")

    report_path = work / "validation-report.json"
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["HF_DATASETS_OFFLINE"] = "1"
    validator = subprocess.run(
        [
            sys.executable,
            "-m",
            "areal.dataset.validate",
            str(dataset_dir),
            "--profile",
            "text-rl",
            "--require-field",
            "answer",
            "--report",
            str(report_path),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validator.returncode == 0, validator.stderr
    assert json.loads(validator.stdout)["checked"] == 3
    validation_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert validation_report["checked"] == 3
    assert validation_report["valid"] == 3
    assert validation_report["invalid"] == 0
    assert not validation_report["diagnostics_truncated"]
```

<!-- areal: offline-dataset-example end -->
