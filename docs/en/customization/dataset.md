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

## Prepare and validate a local dataset offline

Run this example in the prepared Python 3.12+ CPU environment. The neutral directory
names select the generic loader. It reads prepared records as saved. Apply any
tokenization or length filtering before saving them. For text RL, the workflow uses
`messages` as the prompt and passes the row's fields to the reward function, including
the reference `answer` consumed by the GSM8K reward.

```python
from pathlib import Path
from subprocess import run
from sys import executable
from tempfile import TemporaryDirectory

from datasets import Dataset, DatasetDict

from areal.dataset import get_custom_dataset

rows = [
    {"messages": [{"role": "user", "content": question}], "answer": answer}
    for question, answer in [("8 + 3 = ?", "11"), ("14 - 6 = ?", "8"), ("4 * 7 = ?", "28")]
]
with TemporaryDirectory(prefix="local-math-") as root:
    data = Dataset.from_list(rows)
    single = str(Path(root) / "examples")
    data.save_to_disk(single)
    loaded = get_custom_dataset(path=single, type="rl")
    assert len(loaded) == 3
    assert list(loaded) == rows

    splits = str(Path(root) / "splits")
    DatasetDict(train=data.select([0, 1]), validation=data.select([2])).save_to_disk(splits)
    for name, count in [("train", 2), ("validation", 1)]:
        assert len(get_custom_dataset(path=splits, type="rl", split=name)) == count
    try:
        get_custom_dataset(path=splits, type="rl", split="holdout")
    except ValueError as error:
        assert all(name in str(error) for name in ("holdout", "train", "validation"))
    else:
        raise AssertionError("Expected ValueError for missing holdout split")

    run(
        [executable, "-m", "areal.dataset.validate", single,
         "--profile", "text-rl", "--require-field", "answer"],
        check=True,
    )
```

The final command prints `checked=3 valid=3 invalid=0`. For a saved DatasetDict, pass
`--split train` or another available split. JSONL files can be checked directly. The
supported text profiles are:

- `text-rl` checks message objects with a system, user, or assistant role and string
  content. Repeat `--require-field` to require additional keys.
- `tokenized-sft` checks equal non-empty lists of non-negative integer token IDs
  (excluding booleans) and binary loss masks (including booleans), with at least one
  supervised token. Use `--max-length N` to bound sequence length.

Use `--report PATH` for JSON counts and ordered diagnostics with a field, code, and
one-based file line or zero-based dataset row. JSONL is read one line at a time.
`--max-errors N` bounds stored diagnostics (default 20); scanning continues and
`truncated` records omitted details. Exit status is 0 for valid data, 1 for invalid or
empty data, and 2 for usage or I/O errors.
