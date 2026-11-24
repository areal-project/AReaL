# 数据集

**AReaL** 直接集成 HuggingFace `datasets` 包中的 `Dataset` 类。这使您可以在训练前完全灵活地加载、处理和过滤数据。

所需的列名（键）和数据格式取决于 Agent Workflow（用于在线强化学习）或训练引擎（用于离线训练，例如用于监督微调 SFT 的 `LMEngine`）的具体实现。

以下是现有实现中的两个具体示例：

## SFT（离线训练）

在 SFT 示例中，加载的数据直接传递给 `train_lm` 方法：

```python
# areal/trainer/sft_trainer.py
for global_step in range(start_step, max_steps):
    batch = self._load_bcast_from(data_generator)
    self.actor.train_lm(batch)
```

在这种情况下，`train_lm` 方法需要 "input_ids"、"attention_mask" 和 "loss_mask" 键才能工作。我们首先对数据集进行分词以提取
"input_ids" 和 "loss_mask"。然后，使用 `pad_sequences_to_tensors` 方法来批量处理多个序列并附加
"attention_mask"：

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

## GRPO（在线训练）

在 GRPO 示例中，加载的数据首先用于推理而不是训练：

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

请注意，这里的 `collate_fn` 是一个恒等函数，这意味着它只是将各个数据项的列表作为一个批次返回。在 `prepare_batch` 中，数据随后被分派到
Workflow 的多个并发执行中，其中每个分派的数据对应一个单独的 episode。

在以下部分中，我们以
[`RLVRWorkflow`](https://github.com/areal-project/AReaL/blob/main/areal/workflow/rlvr.py)
为例。Agent Workflow 使用输入数据的模式相同。只要符合您的 Workflow 实现，您可以随意修改自定义数据集以包含任何键。

`RLVRWorkflow` 实现从数据字典中提取 "messages" 字段作为生成响应的提示。此外，此数据作为关键字参数传递给
`reward_fn`，这允许奖励函数利用数据集中的其他字段，如 "answers"。示例如下：

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

因此，必须在加载数据集时构建 "messages" 字段，并且奖励函数应该被定义为处理数据集的特定字段。以下是您可以如何为此示例处理数据集：

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

## 自定义本地 RL 数据集：端到端示例

上面的例子展示了内置加载器如何构造记录。本节覆盖约定的另一端：自己编写本地数据集，用 `Dataset.save_to_disk` 保存，再通过 `get_custom_dataset` 加载，无需新增加载器。

### 1. 创建并保存记录

文本 RL 记录需要 `messages` 字段，因为 `RLVRWorkflow` 等 rollout workflow 会把它作为 prompt。额外字段会作为关键字参数传给 reward 函数，因此 `answer` 可以保留在记录中。

```python
from datasets import Dataset, DatasetDict

records = [
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 12 * 7? Return the answer in \\boxed{}.",
            }
        ],
        "answer": "84",
        "task_id": "arith-0001",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "What is 9 + 14? Return the answer in \\boxed{}.",
            }
        ],
        "answer": "23",
        "task_id": "arith-0002",
    },
]

dataset = DatasetDict({"train": Dataset.from_list(records)})
dataset.save_to_disk("/absolute/path/to/arith_rl")
```

### 2. 在配置中指向该目录

```yaml
train_dataset:
  path: /absolute/path/to/arith_rl
  type: rl
  max_length: 2048
```

`path` 必须是 `save_to_disk` 写出的目录，`type: rl` 表示使用文本 RL 契约。已 tokenize 的 SFT 数据请使用 `type: sft`，并保留 `input_ids` 与 `loss_mask` 两列。

### 3. 训练前先加载并检查

```python
from areal.dataset import get_custom_dataset

ds = get_custom_dataset(split="train", path="/absolute/path/to/arith_rl", type="rl")
print(ds)
print(ds.column_names)  # ['messages', 'answer', 'task_id']
print(ds[0]["messages"])
```

### 各数据类型要求的列

| `type` | 必需列 | 使用方 |
| --- | --- | --- |
| `rl` | `messages` | rollout workflow（`RLVRWorkflow`、agent workflow） |
| `sft` | `input_ids`、`loss_mask` | `LMEngine.train_lm`（已 tokenize 的 SFT） |

从磁盘加载的数据集会在加载时按上表校验。缺少列会抛出 `ValueError` 并指明缺少的字段，而不是等到训练器内部才失败。
