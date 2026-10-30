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

## 本地自定义数据集

前面两节展示的都是内置 adapter 如何处理官方数据集。如果您的数据已经准备好，也可以直接落成本地数据集：用 `Dataset.from_list()`
构造记录，`save_to_disk()` 写入目录，再通过 `get_custom_dataset()` 读回来。下面的例子全部在 CPU
上运行，不下载数据集，也不加载模型权重。

文本强化学习所需的字段与 GSM8K adapter 产出的结构一致，即一个用于构造提示的 `messages` 列表，以及一个供奖励函数使用的 `answer` 字段：

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

`\boxed{}` 这条指令与前面 GSM8K adapter 的做法一致，它把这个后缀追加到每个问题末尾，好让奖励函数有明确的目标可抽取。

保存后用统一入口读回：

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

输出如下：

```text
(AReaL) 20261030-16:54:32.139 Dataset WARNING: dataset_config is not provided
columns = ['messages', 'answer']
len     = 3
```

这里有两点需要注意。

第一，那条 WARNING 是预期行为。直接传 `path` 和 `type` 会走到没有 `dataset_config`
对象的分支，加载器记录一条警告后照常工作。在真实的训练配置中，您会设置 `train_dataset.path` 与 `train_dataset.type`，不会看到它。

第二，目录名会影响分发。加载器按路径中的子串选择 adapter，包括
`gsm8k`、`geometry3k`、`clevr_count_70k`、`virl39k`、`hh-rlhf`、`torl_data`、`dapo-math-17k`，以及作为路径片段出现的
`swe`。因此一个叫 `my-gsm8k-copy` 的目录会被交给 GSM8K adapter 然后失败——它期望的是 Hub 上的数据集名，而不是本地目录。示例中的
`math-rl` 是刻意取的中性名字。

还需要强调的是，通用加载路径返回的正是保存时的原始记录：它不做分词，也不应用 `max_length`。这两件事都属于内置 adapter 的行为，它们在返回前会执行
`tokenizer.encode` 并按序列长度过滤。如果需要其中任何一项，请在保存前自行处理，或者参照前两节实现一个 adapter。

至于数据如何进入训练，前文 GRPO 一节已经说明：`RLVRWorkflow` 把 `data["messages"]` 交给 `apply_chat_template`
构造提示，整行记录则以关键字参数展开传给奖励函数。正因如此，`answer` 会作为同名参数抵达 `gsm8k_reward_fn`，由 `math_verify`
与模型输出比对。它从不参与分词，也不会喂给模型；您新增的任何字段都会以同样的方式抵达。

## 校验已准备好的数据

在占用 GPU 之前，可以先用同一个包里的校验器确认数据形状：

```bash
python -m areal.dataset.validate /tmp/areal-offline-tutorial-xxxx/math-rl \
    --profile text-rl --require-field answer
```

```text
checked=3 valid=3 invalid=0
```

除数据集目录外，它也接受 JSONL 文件，当准备脚本写的是逐行记录时更方便：

```bash
python -m areal.dataset.validate samples.jsonl --profile text-rl --max-errors 5
```

| 选项              | 含义                                                        |
| ----------------- | ----------------------------------------------------------- |
| `--profile`       | `text-rl` 或 `tokenized-sft`，必填                          |
| `--split`         | 输入是 `DatasetDict` 时必填                                 |
| `--require-field` | 额外要求该字段存在；可重复（例如 `answer`）                 |
| `--max-length`    | 拒绝超过 N 个 token 的 `tokenized-sft` 记录，仅限该 profile |
| `--max-errors`    | 限制打印多少条诊断（默认 20）                               |
| `--report`        | 把完整结果写成 JSON                                         |

退出码 0 表示每一行都可用，1 表示至少有一行非法或输入为空，2 表示校验器根本读不了输入——路径写错、缺少指定的 split，或 `--report`
的目标路径不可写。诊断信息会指明位置与字段，例如 `line 7: messages[1].role [unsupported_role] ...`。

## 划分训练集与验证集

单个 `Dataset` 没有 split 的概念。想让加载器按名字选取，就另存一个 `DatasetDict`：

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

请求一个不存在的 split 会抛 `ValueError`。加载器会把原始信息包在自己那段受支持数据集列表之后，所以断言时请匹配子串而不是整串：

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
