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

## 离线准备记录

通用加载分支接受已经准备完成并通过 `Dataset.save_to_disk()` 保存的记录。它只从本地磁盘读取这些 记录，不会自动分词、应用 chat
template，也不会按 `max_length` 过滤。对于文本 RL， `RLVRWorkflow` 使用 `messages`
构造生成提示，并把整条记录的字段传给奖励函数。因此在 GSM8K 示例中，`answer` 会到达 `gsm8k_reward_fn`；下面的校验器只检查 `answer`
字段存在，不评价答案是否正确。

校验器支持两种本地 profile。`text-rl` 检查记录是否为对象、`messages` 是否非空、角色是否受支持以及消息内容是否为字符串；
`tokenized-sft` 检查非空的 `input_ids` 与 `loss_mask`、长度是否一致、token ID
是否为非负且非布尔整数、掩码是否为布尔值或精确数值 `0`/`1`，以及是否至少有一个参与监督的位置。它还可以用 `--max-length` 检查序列上限。

JSON 报告包含 `profile`、`checked`、`valid`、`invalid`、`diagnostics_truncated` 和有序的
`diagnostics`。退出码 0 表示输入非空且全部合法；1 表示选定输入为空或存在非法记录；2 表示参数、路径、加载、分区或报告错误。`--max-errors`
只限制保存的详细诊断条数，不限制实际扫描记录数。包含 `train` 与 `validation` 的持久化 `DatasetDict` 必须用 `--split train`
或 `--split validation` 选择一个分区。JSONL 按物理行流式读取，内存受最大单行、运行库和保存的诊断条数影响，而不是整个文件；Hugging Face
目录使用本地 `datasets` Arrow loader，其内存行为由该库决定。

内置 adapter 匹配会检查完整路径，包括父目录。请让准备数据的目录及其父目录避开 `gsm8k`、`geometry3k` 等名称。下面的临时目录前缀已避开这些名称。

请在 AReaL 仓库根目录运行以下完整示例。它会创建三条本地数学记录，分别保存 `Dataset` 与 `DatasetDict`，通过现有 loader
重新加载，检查显式分区和缺失分区错误，并在临时目录删除前调用真实的校验模块。

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
