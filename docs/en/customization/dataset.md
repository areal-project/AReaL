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

## Offline Data Preparation: A Complete Custom Dataset Example

GPU availability is often the bottleneck for new experiments. This section shows how to
create, save, load, and check a small custom RL dataset entirely on CPU, using the
generic load-from-disk fallback that AReaL uses when a dataset path does not match any
built-in adapter. The example runs in a prepared Python 3.12+ environment: it never
downloads a dataset or model, starts no inference server, and initializes no distributed
training.

### Create and save prepared records

Text RL workflows such as `RLVRWorkflow` read the `messages` field as the prompt, and
`gsm8k_reward_fn` reads the `answer` field to score completions. Three synthetic math
samples below already satisfy that contract, so they can be prepared without any
tokenization step:

```python
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset

samples = [
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Nina has 3 apples and buys 4 more. How many apples "
                    "does she have? Put your final answer within \\boxed{}."
                ),
            }
        ],
        "answer": "7",
    },
    {
        "messages": [
            {"role": "system", "content": "You are a careful math tutor."},
            {
                "role": "user",
                "content": (
                    "A book costs $12 and a pen costs $5. How much more "
                    "does the book cost? \\boxed{}"
                ),
            },
        ],
        "answer": "7",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Tom rides his bike for 2 hours at 10 km/h. How far "
                    "does he travel (in km)? \\boxed{}"
                ),
            }
        ],
        "answer": "20",
    },
]

data_root = Path(tempfile.mkdtemp(prefix="areal_math_demo_"))
data_dir = data_root / "my_math_data"

Dataset.from_list(samples).save_to_disk(str(data_dir))
```

The directory name matters for dispatching: `_get_custom_dataset()` matches built-in
adapters by substrings such as `gsm8k` or path tokens such as `swe` (see
`areal/dataset/__init__.py`). A neutral name like `my_math_data` does not trigger any
adapter, so the loader falls through to the generic Hugging Face `load_from_disk()` path
— exactly what training will do for your own prepared data.

### Load through `get_custom_dataset()`

```python
from areal.api.cli_args import TrainDatasetConfig
from areal.dataset import get_custom_dataset

dataset_config = TrainDatasetConfig(path=str(data_dir), type="rl")
loaded = get_custom_dataset(dataset_config=dataset_config)

assert len(loaded) == 3
for row, sample in zip(loaded, samples):
    assert row["messages"] == sample["messages"]
    assert row["answer"] == sample["answer"]
```

`get_custom_dataset()` returns exactly the records you saved. At training time,
`RLVRWorkflow` renders `data["messages"]` with
`tokenizer.apply_chat_template(..., add_generation_prompt=True)` to build the prompt,
and passes the remaining fields to the reward function as keyword arguments:
`gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer)` compares the
model completion against `answer`. A custom record therefore only needs to provide the
fields your workflow and reward function read.

One caveat: the generic load-from-disk fallback expects *prepared* records. It does not
tokenize conversations, apply a chat template, or enforce `max_length` — filtering by
`max_length` happens inside built-in adapter loaders (such as `get_gsm8k_rl_dataset()`),
not automatically. If your records might be too long, shorten them during preparation or
pre-tokenize them offline.

### Select named splits from a `DatasetDict`

Datasets saved with `save_to_disk()` may be a `DatasetDict` (several named splits in one
directory). Pass `split=` to select one:

```python
from datasets import DatasetDict

dict_dir = data_root / "my_math_split_data"
DatasetDict(
    {
        "train": Dataset.from_list(samples[:2]),
        "validation": Dataset.from_list(samples[2:]),
    }
).save_to_disk(str(dict_dir))

train = get_custom_dataset(
    split="train",
    dataset_config=TrainDatasetConfig(path=str(dict_dir), type="rl"),
)
validation = get_custom_dataset(
    split="validation",
    dataset_config=TrainDatasetConfig(path=str(dict_dir), type="rl"),
)
assert len(train) == 2
assert len(validation) == 1

try:
    get_custom_dataset(
        split="test",
        dataset_config=TrainDatasetConfig(path=str(dict_dir), type="rl"),
    )
except ValueError as exc:
    message = str(exc)
    assert "'test'" in message  # the requested split is named
    assert "validation" in message  # the available splits are listed

shutil.rmtree(data_root)  # done with the demo data
```

Asking for a split that does not exist raises a `ValueError` identifying both the
requested and the available splits.

### Check prepared data with the validation CLI

Before arranging GPU resources, run the standalone validator over the same data. It
reads local JSONL files or `save_to_disk()` directories on CPU:

```bash
# A single saved dataset (as produced by Dataset.save_to_disk()).
python -m areal.dataset.validate ./my_math_data \
    --profile text-rl --require-field answer

# A DatasetDict: an explicit --split is required, and a JSON report is kept.
python -m areal.dataset.validate ./my_math_split_data \
    --profile text-rl --split validation --require-field answer \
    --report validation_report.json
```

`--profile` selects which downstream contract to check:

| Profile         | Requires                                                                                                                                                                             | Notes                                                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `text-rl`       | non-empty `messages` list of message objects with `role` in `system`/`user`/`assistant` and string `content`                                                                         | extra fields are allowed; repeatable `--require-field` checks the presence of reward-specific fields such as `answer` (presence only, values are not scored) |
| `tokenized-sft` | non-empty, equally sized `input_ids` and `loss_mask` lists; non-negative integer token IDs (booleans rejected); binary `0`/`1` or boolean mask values; at least one supervised token | `--max-length N` (this profile only) flags records with more than N tokens                                                                                   |

The profiles describe supported subsets of these contracts, not universal AReaL schemas:
remote downloads, tokenization, multimodal payloads, tool-call conversations, and DPO
pairs are outside the validator's scope.

Exit codes: `0` means every record is valid, `1` means at least one record is invalid or
the input contains no records at all, and `2` means a usage, input-loading, or
report-writing failure.

#### Design notes

- **Report format.** `--report` writes a deterministic JSON object with `checked` /
  `valid` / `invalid` record counts and ordered `diagnostics`; each diagnostic has a
  `location` (`line N`, 1-based, for JSONL files or `row N`, 0-based, for dataset rows),
  a `field` path such as `messages[0].role`, a stable `code` such as `role_invalid`, and
  a `message` with the concrete value. Repeated runs over unchanged data produce the
  same counts and the same ordered diagnostics.
- **Diagnostic truncation.** Detailed diagnostics are capped by `--max-errors` (default
  10), but the scan continues over every record and the report counts stay complete;
  when any diagnostic is dropped, the report sets `"truncated": true`.
- **Memory behavior.** JSONL files are streamed line by line, so validation memory stays
  bounded regardless of file size; saved datasets are scanned row by row. Neither path
  writes to or modifies the input.

Pairing this section with the reward-function contract above, the offline loop for a new
custom dataset is: prepare records → `save_to_disk()` → reload with
`get_custom_dataset()` → `python -m areal.dataset.validate` → only then arrange GPUs.
