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

## 离线数据准备：完整自定义数据集示例

GPU 资源往往是新实验的瓶颈。本节演示如何完全在 CPU 上创建、保存、加载并检查 一个小型自定义 RL 数据集，使用的是当数据集路径不匹配任何内置适配器时 AReaL
采用的通用 load-from-disk 回退路径。示例在已配置好的 Python 3.12+ 环境中即可
运行：不会下载任何数据集或模型、不启动推理服务器、也不初始化分布式训练。

### 创建并保存已准备好的数据记录

`RLVRWorkflow` 等文本 RL Workflow 会读取 `messages` 字段作为提示词，而 `gsm8k_reward_fn` 会读取 `answer`
字段为生成结果打分。下面三个合成的数学样本 已经满足该契约，因此无需任何分词步骤即可直接准备：

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

目录名会影响加载派发：`_get_custom_dataset()` 通过诸如 `gsm8k` 的子串或 `swe` 这样的路径标记匹配内置适配器（见
`areal/dataset/__init__.py`）。像 `my_math_data` 这样中性的名称不会触发任何适配器，加载器会落到通用的 Hugging Face
`load_from_disk()` 路径——训练时加载您自己准备的数据也正是这条路径。

### 通过 `get_custom_dataset()` 加载

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

`get_custom_dataset()` 返回的正是您保存的记录。训练时，`RLVRWorkflow` 使用
`tokenizer.apply_chat_template(..., add_generation_prompt=True)` 渲染 `data["messages"]`
构造提示词，并把其余字段作为关键字参数传给奖励函数：
`gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer)` 会将模型生成结果与
`answer` 对比打分。因此自定义记录只需提供您的 Workflow 与奖励函数实际读取的字段。

一个注意事项：通用的 load-from-disk 回退路径期望的是*已准备好*的记录。它不会 对对话分词、不会应用 chat template、也不会强制
`max_length`——按 `max_length` 过滤发生在内置适配器加载函数（如 `get_gsm8k_rl_dataset()`）内部，而不是自动
进行。如果记录可能过长，请在准备阶段缩短它们，或离线预先分词。

### 从 `DatasetDict` 中选择命名 split

用 `save_to_disk()` 保存的数据集可能是 `DatasetDict`（一个目录中包含多个命名 split）。传入 `split=` 即可选择其中一个：

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
    assert "'test'" in message  # 错误信息指出所请求的 split
    assert "validation" in message  # 并列出所有可用的 split

shutil.rmtree(data_root)  # 演示数据用完即可删除
```

请求不存在的 split 会抛出 `ValueError`，同时指出所请求的 split 和所有可用的 split。

### 使用验证 CLI 检查已准备的数据

在申请 GPU 资源之前，先对同一份数据运行独立的验证器。它在 CPU 上读取本地 JSONL 文件或 `save_to_disk()` 目录：

```bash
# 单个已保存数据集（由 Dataset.save_to_disk() 产生）。
python -m areal.dataset.validate ./my_math_data \
    --profile text-rl --require-field answer

# DatasetDict：必须显式传入 --split，并保留一份 JSON 报告。
python -m areal.dataset.validate ./my_math_split_data \
    --profile text-rl --split validation --require-field answer \
    --report validation_report.json
```

`--profile` 选择要检查的下游契约：

| Profile         | 要求                                                                                                                                            | 说明                                                                                                 |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `text-rl`       | 非空 `messages` 列表，元素为消息对象，`role` 必须是 `system`/`user`/`assistant` 之一，`content` 必须是字符串                                    | 允许额外字段；可重复的 `--require-field` 检查奖励相关字段（如 `answer`）的存在性（只查存在，不评分） |
| `tokenized-sft` | 非空且等长的 `input_ids` 与 `loss_mask` 列表；token id 为非负整数（不接受布尔值）；mask 取值为二进制的 `0`/`1` 或布尔值；至少包含一个监督 token | `--max-length N`（仅适用于该 profile）会标记超过 N 个 token 的记录                                   |

这些 profile 描述的是上述契约的受支持子集，而不是通用的 AReaL schema：远程 下载、分词、多模态载荷、工具调用对话与 DPO 数据对都不在验证器范围内。

退出码：`0` 表示所有记录合法；`1` 表示存在非法记录，或输入中没有任何记录； `2` 表示用法错误、输入加载失败或报告写入失败。

#### 设计说明

- **报告格式。** `--report` 写出确定性的 JSON 对象，包含 `checked` / `valid` / `invalid` 记录计数与有序
  `diagnostics`；每条诊断包含 `location`（JSONL 文件为 1 起始的 `line N`，数据集行为 0 起始的 `row N`）、诸如
  `messages[0].role` 的 `field` 路径、诸如 `role_invalid` 的稳定 `code`，以及 带具体值的
  `message`。对未变化的数据重复运行会得到相同的计数与相同顺序的诊 断。
- **诊断截断。** 详细诊断数量由 `--max-errors`（默认 10）限制，但扫描会遍历 全部记录、计数保持完整；一旦有诊断被丢弃，报告会置
  `"truncated": true`。
- **内存行为。** JSONL 文件按行流式读取，验证内存占用与文件大小无关；已保存 数据集按行遍历。两条路径都不会写入或修改输入数据。

结合上文奖励函数契约，新自定义数据集的离线闭环是：准备记录 → `save_to_disk()` → 用 `get_custom_dataset()` 重新加载 → 运行
`python -m areal.dataset.validate` → 一切通过后再申请 GPU。
