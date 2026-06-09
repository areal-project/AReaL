# 奖励函数

本指南介绍如何在 AReaL 中自定义强化学习所用的奖励函数。AReaL 支持**基于规则的奖励函数**和
**生成式（LLM-as-Judge）奖励模型**。

## 奖励函数签名

所有与 `RLVRWorkflow` 配合使用的奖励函数遵循以下签名：

```python
def my_reward_fn(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    **kwargs,
) -> float:
    """
    Args:
        prompt: 输入的提示文本。
        completions: 模型生成的回复文本。
        prompt_ids: 提示的 token ID 列表。
        completion_ids: 回复的 token ID 列表。
        **kwargs: 数据集中的其他字段（如 answer、solution 等）。

    Returns:
        奖励值（float 标量）。
    """
```

`**kwargs` 会接收数据集中的所有其他字段。例如，如果数据集中包含 `"answer"` 列，
你可以通过 `kwargs["answer"]` 访问，或直接将 `answer` 作为命名参数。

## 基于规则的奖励函数

最简单的奖励函数通过规则将模型输出与标准答案进行对比。

### 示例：数学答案验证

```python
# my_project/rewards.py
from areal.reward import get_math_verify_worker

def my_math_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """使用 AReaL 内置的数学验证器检验答案。"""
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        return 0.0
```

内置的 `MathVerifyWorker` 支持 LaTeX 和表达式提取，可配置精度。
详见 `areal/reward/__init__.py`。

### 示例：格式 + 准确率复合奖励

```python
import re

def composite_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    """结合格式合规性和答案准确性的奖励函数。"""
    format_score = 0.0
    accuracy_score = 0.0

    # 检查回复是否遵循预期格式（如使用 \boxed{}）
    if re.search(r"\\boxed\{.+\}", completions):
        format_score = 0.2

    # 检查答案准确性
    from areal.reward import get_math_verify_worker
    try:
        accuracy_score = get_math_verify_worker().verify(
            str(completions), str(answer)
        ) * 0.8
    except Exception:
        pass

    return format_score + accuracy_score
```

## 内置奖励函数

AReaL 内置以下奖励函数：

| 函数 | 模块路径 | 数据集 |
|------|---------|--------|
| `gsm8k_reward_fn` | `areal.reward.gsm8k.gsm8k_reward_fn` | GSM8K 数学 |
| `geometry3k_reward_fn` | `areal.reward.geometry3k.geometry3k_reward_fn` | Geometry3K |
| `clevr_count_70k_reward_fn` | `areal.reward.clevr_count_70k.clevr_count_70k_reward_fn` | CLEVR Count |

## 在训练中使用奖励函数

通过**模块路径字符串**将奖励函数传递给 workflow，这允许 AReaL 在分布式 worker 间
序列化和分发该函数：

```python
from areal import PPOTrainer

workflow_kwargs = dict(
    reward_fn="my_project.rewards.my_math_reward_fn",  # 可导入的字符串路径
    gconfig=config.gconfig,
    tokenizer=config.tokenizer_path,
)

with PPOTrainer(config, train_dataset=train_dataset) as trainer:
    trainer.train(
        workflow="areal.workflow.rlvr.RLVRWorkflow",
        workflow_kwargs=workflow_kwargs,
    )
```

也可以直接传递函数对象（在不需要分布式序列化时可用）：

```python
from my_project.rewards import my_math_reward_fn

workflow_kwargs = dict(
    reward_fn=my_math_reward_fn,  # 直接函数引用
    gconfig=config.gconfig,
    tokenizer=tokenizer,
)
```

## 生成式奖励模型（LLM-as-Judge）

对于规则验证不足以覆盖的任务（如开放式生成、创意写作、指令遵循），
可以使用**生成式奖励模型**，通过提示另一个 LLM 来对回复进行评分。

### 示例：LLM-as-Judge 奖励函数

```python
import re

def llm_judge_reward_fn(
    prompt, completions, prompt_ids, completion_ids, **kwargs
) -> float:
    """使用外部 LLM API 评判回复质量。"""
    import openai

    judge_prompt = f"""请对以下回复进行评分，评分范围 0-10。
只输出数字分数。

问题：{prompt}
回复：{completions}

分数："""

    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",  # 本地 vLLM/SGLang 服务
        api_key="unused",
    )
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0.0,
        max_tokens=16,
    )
    score_text = response.choices[0].message.content.strip()

    # 提取数字分数
    match = re.search(r"(\d+(?:\.\d+)?)", score_text)
    if match:
        score = float(match.group(1))
        return min(score / 10.0, 1.0)  # 归一化到 [0, 1]
    return 0.0
```

### 处理慢速奖励函数

生成式奖励模型计算可能较慢。AReaL 的 `AsyncRewardWrapper` 自动将你的奖励函数包装为
异步执行，并提供超时处理：

```python
from areal.api import AsyncRewardWrapper

# 当你将 reward_fn 传递给 RLVRWorkflow 时，包装器会自动应用。
# 自定义超时和并发数：
async_reward = AsyncRewardWrapper(
    reward_fn=llm_judge_reward_fn,
    timeout_seconds=30,   # 对慢速模型增加超时（默认 15 秒）
    max_workers=4,        # 并行奖励计算的进程数
    max_retries=3,        # 崩溃时自动恢复
)
```

`RLVRWorkflow` 会自动使用 `AsyncRewardWrapper` 包装你的奖励函数，
通常无需手动创建。默认超时为 15 秒——如果你的奖励函数更慢，
请考虑对其进行优化或批量处理请求。

## 注册新的内置奖励函数

要将奖励函数添加到 AReaL 的内置集合中（使其可按数据集名称自动选择）：

1. 在 `areal/reward/my_dataset.py` 创建奖励模块
2. 在 `areal/reward/__init__.py` 中注册：

```python
# 在 areal/reward/__init__.py 中
VALID_REWARD_FN = ["clevr_count_70k", "geometry3k", "my_dataset"]

def get_custom_reward_fn(path: str, **kwargs):
    # ... 已有的条目 ...
    elif "my_dataset" in path:
        from .my_dataset import my_dataset_reward_fn
        return my_dataset_reward_fn
```

详见 [Add Reward 技能指南](https://github.com/inclusionAI/AReaL/blob/main/.claude/skills/add-reward/SKILL.md)
了解完整步骤。
