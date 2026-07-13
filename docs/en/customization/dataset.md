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
