# 自定义奖励函数

本指南介绍如何在 AReaL 中创建用于 RL 训练的自定义奖励函数。

## 奖励函数签名

每个奖励函数遵循相同的签名：

```python
def my_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs,
) -> float:
    """
    Parameters
    ----------
    prompt : str
        解码后的提示文本。
    completions : str
        模型生成的补全文本。
    prompt_ids : list[int]
        提示的 token ID。
    completion_ids : list[int]
        补全的 token ID。
    **kwargs
        数据集中的额外字段（如 "answer"、"solution"）。

    Returns
    -------
    float
        标量奖励值。
    """
    ...
```

`**kwargs` 接收数据集中的所有额外列。例如，如果您的数据集有 `answer` 字段，可以通过 `kwargs["answer"]` 访问。

## 基于规则的奖励

最简单的方法是基于规则的奖励，将模型输出与已知答案进行比对。

### 示例：精确匹配

```python
def exact_match_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    return 1.0 if answer.strip() in completions else 0.0
```

### 示例：数学验证

AReaL 内置了使用 `math-verify` 库的数学奖励函数：

```python
# areal/reward/gsm8k.py
from areal.reward import get_math_verify_worker

def gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        return 0.0
```

## 基于模型的奖励（Critic-Like）

对于没有明确答案的任务，可以使用**预训练的奖励模型**来评分补全内容。典型的 critic-like 奖励模型是一个序列分类模型（例如使用
[`examples/alignment/`](https://github.com/inclusionAI/AReaL/tree/main/examples/alignment/)
训练的 Bradley-Terry 模型），输入 (prompt, response)，输出标量分数。

### 示例：使用预训练奖励模型

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_model = None
_tokenizer = None

def my_reward_model_fn(prompt, completions, prompt_ids, completion_ids, **kwargs):
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("my-org/my-reward-model")
        _model = AutoModelForSequenceClassification.from_pretrained(
            "my-org/my-reward-model", torch_dtype=torch.bfloat16
        ).cuda().eval()

    inputs = _tokenizer(
        prompt + completions,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to("cuda")

    with torch.no_grad():
        score = _model(**inputs).logits.squeeze().float().item()
    return score
```

**要点：**

- **懒加载**：只加载模型一次，并全局缓存。奖励函数在每个训练步骤中被调用多次，避免每次都重新加载。
- **GPU 放置**：使用 `torch.bfloat16` 和 `.cuda()` 提高效率。奖励模型与 rollout worker 运行在同一 GPU 上。
- **错误处理**：将函数体包裹在 try/except 中，失败时返回 0.0，避免训练循环崩溃。

## 生成式奖励（LLM-as-Judge）

对于难以通过编程方式验证正确性的任务，可以使用**生成式模型作为评判者** —— 一个独立的 LLM 对策略模型的输出进行评估并产生奖励分数。

AReaL 在
[Tongyi DeepResearch 搜索智能体示例](https://github.com/inclusionAI/AReaL/tree/main/examples/search_agent/tongyi_deepresearch/)中包含了完整的
LLM-as-judge 实现。该模式包含三个核心组件：

### 1. 评判提示模板

评判者接收问题、标准答案和模型预测，然后输出结构化的 JSON 判定：

````python
# 来自 examples/search_agent/tongyi_deepresearch/react_agent.py
judge_prompt_template = (
    "You are an evaluation assistant. Please determine if the predicted answer "
    "is equivalent to the labeled answer.\n"
    "question: {question}\n"
    "ground truth answers: {gt_answer}\n"
    "pred_answer: {pred_answer}\n\n"
    "The output should in the following json format:\n"
    '```json\n{{\n'
    '    "rationale": "your rationale",\n'
    '    "judgement": "correct or incorrect"\n'
    '}}\n```\n'
)
````

### 2. 解析评判结果

````python
# 来自 examples/search_agent/tongyi_deepresearch/react_agent.py
import ast, json

def parse_judge_result(raw_response):
    mbe = None
    for parse_fn in [json.loads, ast.literal_eval]:
        try:
            mbe = parse_fn(
                raw_response.split("```json")[-1].split("```")[0].strip()
            )
            break
        except Exception:
            pass
    if mbe is None and '"judgement": "correct"' in raw_response:
        mbe = dict(judgement="correct")
    if mbe is None:
        mbe = dict(judgement="unknown")
    return float("judgement" in mbe and mbe["judgement"] == "correct")
````

### 3. 通过独立推理引擎调用评判者

与 critic-like 奖励不同，生成式评判者需要自己的推理引擎。这在自定义工作流内完成，而非作为独立的 `reward_fn`：

```python
# 来自 examples/search_agent/tongyi_deepresearch/train.py
class MyJudgeWorkflow(RolloutWorkflow):
    def __init__(self, ..., judge_engine_config=None):
        # 为评判者创建独立推理引擎
        if judge_engine_config is not None:
            self.judge_engine = RemoteSGLangEngine(judge_engine_config)
            self.judge_engine.initialize()
        self.judge_client = ArealOpenAI(
            engine=self.judge_engine, tokenizer=tokenizer
        )
```

评判者在采集轨迹时异步调用：

```python
async def calc_reward_with_llm_judge(self, result):
    judge_prompt = judge_prompt_template.format(
        question=result["question"],
        gt_answer=str(result["answer"]),
        pred_answer=result["prediction"][:200],
    )
    completion = await self.judge_client.chat.completions.create(
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=1.0,
        max_completion_tokens=8192,
        store=False,  # 不将评判 token 记录为训练数据
    )
    return parse_judge_result(completion.choices[0].message.content)
```

**为什么需要独立引擎？** 生成式评判者需要 GPU 推理，无法在 `AsyncRewardWrapper` 使用的 `ProcessPoolExecutor`
中运行。因此，它们作为自定义工作流的一部分，使用专用推理引擎运行。

完整的工作示例请参见
[`examples/search_agent/tongyi_deepresearch/`](https://github.com/inclusionAI/AReaL/tree/main/examples/search_agent/tongyi_deepresearch/)。

## 异步奖励包装器

对于可能较慢的奖励函数（如调用外部 API、运行大量计算），AReaL 提供 `AsyncRewardWrapper`，在进程池中运行并处理超时：

```python
from areal.api.reward_api import AsyncRewardWrapper

# 将同步奖励函数包装为异步执行
async_reward = AsyncRewardWrapper(
    my_reward_fn,
    timeout_seconds=15,   # 计算超过 15 秒则返回 0.0
    max_workers=4,        # 进程池大小
    max_retries=3,        # 失败时重试次数
)

# 在工作流中使用：
reward = await async_reward(prompt, completions, prompt_ids, completion_ids, **data)
```

内置的 `RLVRWorkflow` 会自动用 `AsyncRewardWrapper` 包装您的奖励函数。

## 与训练集成

将奖励函数作为字符串导入路径传递给训练器：

```python
from areal import PPOTrainer

with PPOTrainer(config, train_dataset=train_dataset) as trainer:
    trainer.train(
        workflow="areal.workflow.rlvr.RLVRWorkflow",
        workflow_kwargs=dict(
            reward_fn="my_module.my_reward_fn",  # 字符串导入路径
            gconfig=config.gconfig,
            tokenizer=config.tokenizer_path,
        ),
    )
```

字符串导入路径允许奖励函数被序列化并分发到 worker 进程。

## 另请参阅

- [RLHF 奖励建模示例](https://github.com/inclusionAI/AReaL/tree/main/examples/alignment/) - 训练
  Bradley-Terry 奖励模型
- [LLM-as-Judge 示例](https://github.com/inclusionAI/AReaL/tree/main/examples/search_agent/tongyi_deepresearch/)
  \- 使用独立评判引擎的生成式奖励
- [异步工作流最佳实践](../best_practices/workflow.md) - 编写高效的异步奖励函数
