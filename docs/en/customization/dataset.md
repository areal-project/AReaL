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

## Custom datasets on disk

The two examples above show how the built-in adapters consume official datasets. If your
records are already prepared, you can skip the adapter entirely: build them with
`Dataset.from_list()`, persist them with `save_to_disk()`, and read them back through
`get_custom_dataset()`. Everything below runs on CPU, without downloading a dataset or
loading model weights.

For text RL the required shape matches what the GSM8K adapter produces: a `messages`
list that becomes the prompt, and an `answer` field for the reward function.

```python
MATH_SAMPLES = [
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Natalia sold 48 clips in April and half as many clips in May. "
                    "How many clips did Natalia sell altogether in April and May?\n"
                    "Please put your final answer within \\boxed{}."
                ),
            }
        ],
        "answer": "She sold 48 + 24 = 72 clips altogether.\n\\boxed{72}",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Weng earns $12 an hour for babysitting. Yesterday, she just did "
                    "50 minutes of babysitting. How much did she earn?\n"
                    "Please put your final answer within \\boxed{}."
                ),
            }
        ],
        "answer": "12 * (50 / 60) = 10.\n\\boxed{10}",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Betty is saving money for a new wallet which costs $100. Betty has "
                    "only half of the money she needs. Her parents decided to give her "
                    "$15 for that purpose. How much more does she need?\n"
                    "Please put your final answer within \\boxed{}."
                ),
            }
        ],
        "answer": "100 - 50 - 15 = 35.\n\\boxed{35}",
    },
]
```

The `\boxed{}` instruction matches what the GSM8K adapter appends to every question, so
the reward function has an unambiguous target to extract.

Save the records and read them back through the shared entry point:

```python
import tempfile
from pathlib import Path

from datasets import Dataset

from areal.dataset import get_custom_dataset

workdir = Path(tempfile.mkdtemp(prefix="areal-offline-tutorial-"))
dataset_dir = workdir / "math-rl"

Dataset.from_list(MATH_SAMPLES).save_to_disk(str(dataset_dir))
reloaded = get_custom_dataset(path=str(dataset_dir), type="rl")

assert len(reloaded) == 3
assert reloaded[0]["messages"] == MATH_SAMPLES[0]["messages"]
assert reloaded[0]["answer"] == MATH_SAMPLES[0]["answer"]
print("columns =", reloaded.column_names)
print("len     =", len(reloaded))
```

The output is:

```text
(AReaL) 20261030-16:54:32.139 Dataset WARNING: dataset_config is not provided
columns = ['messages', 'answer']
len     = 3
```

Two details are worth knowing.

The warning is expected. Passing `path` and `type` as keyword arguments takes the branch
that has no `dataset_config` object, so the loader logs a warning and carries on. In a
real training config you would set `train_dataset.path` and `train_dataset.type`
instead, and never see it.

The directory name affects dispatch. The loader selects an adapter from substrings in
the path: `gsm8k`, `geometry3k`, `clevr_count_70k`, `virl39k`, `hh-rlhf`, `torl_data`,
`dapo-math-17k`, and `swe` as a path token. A directory called `my-gsm8k-copy` would
therefore be handed to the GSM8K adapter and fail, because that adapter expects a Hub
dataset name rather than a local directory. `math-rl` is deliberately neutral.

It is also worth stressing that the generic path returns rows exactly as they were
saved: it does not tokenize, and it does not apply `max_length`. Both are adapter
behaviours, performed before an adapter returns. If you need either, do it yourself
before saving, or write an adapter as shown in the previous two sections.

As for how the records reach training, the GRPO section above already covers it:
`RLVRWorkflow` passes `data["messages"]` to `apply_chat_template` to build the prompt,
and spreads the whole row into the reward function as keyword arguments. That is why
`answer` arrives as a parameter of the same name in `gsm8k_reward_fn`, where
`math_verify` compares it against the model's completion. It is never tokenized and
never fed to the model, and any field you add reaches the reward function the same way.

## Validating prepared data

Before spending GPUs, the same package can check that the shape is what the trainer will
expect:

```bash
python -m areal.dataset.validate /tmp/areal-offline-tutorial-xxxx/math-rl \
    --profile text-rl --require-field answer
```

```text
checked=3 valid=3 invalid=0
```

Besides dataset directories it also accepts JSONL files, which is convenient when your
preparation script writes lines rather than a saved dataset:

```bash
python -m areal.dataset.validate samples.jsonl --profile text-rl --max-errors 5
```

| Option            | Meaning                                                      |
| ----------------- | ------------------------------------------------------------ |
| `--profile`       | `text-rl` or `tokenized-sft`; required                       |
| `--split`         | required when the input is a `DatasetDict`                   |
| `--require-field` | also demand this field exists; repeatable (e.g. `answer`)    |
| `--max-length`    | reject `tokenized-sft` rows longer than N; that profile only |
| `--max-errors`    | cap how many diagnostics are printed (default 20)            |
| `--report`        | write the full result as JSON                                |

Exit code `0` means every row is usable, `1` means at least one row is invalid or the
input was empty, and `2` means the tool could not read the input at all — a typo in the
path, a missing split, or an unwritable `--report` target. Diagnostics name the location
and the field, for example `line 7: messages[1].role [unsupported_role] ...`.

## Splitting into train and validation

A single `Dataset` has no concept of splits. Save a `DatasetDict` when you want the
loader to select one by name:

```python
from datasets import Dataset, DatasetDict

split_dir = workdir / "math-rl-split"
DatasetDict(
    {
        "train": Dataset.from_list(MATH_SAMPLES[:2]),
        "validation": Dataset.from_list(MATH_SAMPLES[2:]),
    }
).save_to_disk(str(split_dir))

train = get_custom_dataset(path=str(split_dir), type="rl", split="train")
validation = get_custom_dataset(path=str(split_dir), type="rl", split="validation")

assert (len(train), len(validation)) == (2, 1)
```

Asking for a split that is not there raises `ValueError`. The loader wraps the original
message behind its own supported-dataset list, so match on substrings rather than the
whole string:

```python
try:
    get_custom_dataset(path=str(split_dir), type="rl", split="test")
except ValueError as exc:
    message = str(exc)
    assert "test" in message
    assert "train" in message and "validation" in message
```

```text
Dataset /tmp/areal-offline-tutorial-xxxx/math-rl-split with split test and
training type rl is not supported. Supported datasets are: ['gsm8k',
'clevr_count_70k', 'geometry3k', 'virl39k', 'hh-rlhf', 'torl_data',
'dapo-math-17k', 'swe_sft']. Also failed to load from disk: Requested split
'test' not found in DatasetDict at /tmp/areal-offline-tutorial-xxxx/math-rl-split.
Available splits: ['train', 'validation']
```
