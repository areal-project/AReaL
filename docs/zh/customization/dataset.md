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

## 离线准备并校验本地数据集

在准备好的 Python 3.12+ CPU 环境中运行以下示例。临时目录名避开了内置适配器的匹配规则。通用 loader
按保存时的内容读回记录，需要分词或按长度过滤时，请在保存前完成。文本 RL 工作流使用 `messages` 构造提示词，并将整行字段传给奖励函数，其中 `answer` 是
GSM8K 奖励函数使用的参考答案。

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

最后一条命令输出 `checked=3 valid=3 invalid=0`。校验落盘的 DatasetDict 时，通过 `--split train` 等参数选择已有的
split。也可以直接传入 JSONL 文件。支持的文本数据子集如下。

- `text-rl` 检查消息对象的角色（system、user、assistant）和字符串内容。可重复使用 `--require-field` 要求额外字段存在。
- `tokenized-sft` 检查等长且非空的 token ID 和 loss mask 列表。ID 为非负整数且排除布尔值，mask
  接受二值及布尔值，至少有一个位置参与监督。可用 `--max-length N` 限制序列长度。

使用 `--report PATH` 保存 JSON 计数和有序诊断。每条诊断包括字段、错误码及位置。文件行号从 1 开始，落盘数据集行号从 0 开始。程序逐行读取
JSONL。`--max-errors N` 限制保留的诊断数（默认 20），程序继续扫描，`truncated` 标明有无省略的明细。退出码 0 表示数据有效，1
表示数据无效或为空，2 表示用法或读写错误。
