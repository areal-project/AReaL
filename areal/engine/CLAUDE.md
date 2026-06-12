[根目录](../../CLAUDE.md) > **areal/engine**

# areal.engine - 训练与推理引擎

## 变更记录 (Changelog)

### 2026-01-31 - 初始化

- 模块文档创建
- 识别 FSDP、Megatron、SGLang、vLLM 四大引擎

---

## 模块职责

实现训练与推理引擎的适配器：

- **训练引擎**：FSDP2、Megatron Core
- **推理引擎**：SGLang、vLLM（远程服务器）
- **专用引擎**：PPO Actor/Critic、SFT LM、Reward Model

## 入口与启动

引擎通过 `areal.controller` 调用，不直接启动。

## 对外接口

### 训练引擎

| 类名              | 文件                  | 职责                                   |
| ----------------- | --------------------- | -------------------------------------- |
| `FSDPEngine`      | `fsdp_engine.py`      | PyTorch FSDP2 训练引擎（核心）         |
| `MegatronEngine`  | `megatron_engine.py`  | Megatron Core 训练引擎（核心）         |

### 推理引擎

| 类名                  | 文件                  | 职责                                   |
| --------------------- | --------------------- | -------------------------------------- |
| `SGLangRemoteEngine`  | `sglang_remote.py`    | SGLang 远程推理引擎（核心）            |
| `VLLMRemoteEngine`    | `vllm_remote.py`      | vLLM 远程推理引擎（核心）              |

### 专用引擎

| 类名              | 文件                  | 职责                                   |
| ----------------- | --------------------- | -------------------------------------- |
| `PPOActor`        | `ppo/actor.py`        | PPO Actor 引擎                         |
| `PPOCritic`       | `ppo/critic.py`       | PPO Critic 引擎                        |
| `LMEngine`        | `sft/lm_engine.py`    | SFT 语言模型引擎                       |
| `RWEngine`        | `rw/rw_engine.py`     | Reward Model 引擎                      |

## 关键依赖与配置

### 外部依赖

- `torch`：张量操作与分布式通信
- `torch.distributed.fsdp`：FSDP2
- `megatron.core`：Megatron Core
- `sglang`：SGLang 推理引擎
- `vllm`：vLLM 推理引擎
- `transformers`：模型加载与 Tokenizer

### 内部依赖

- `areal.api.engine_api`：`TrainEngine`、`InferenceEngine` 基类
- `areal.api.cli_args`：配置数据类
- `areal.utils.fsdp`：FSDP 工具（检查点、优化器、并行）
- `areal.utils.mcore`：Megatron Core 工具
- `areal.models`：模型实现

## 数据模型

### FSDPEngine 初始化参数

```python
FSDPEngine(
    model_path: str,                    # 模型路径
    parallel_strategy: ParallelStrategy, # 并行策略
    dtype: torch.dtype,                 # 数据类型
    use_lora: bool = False,             # 是否使用 LoRA
    # ... 其他参数
)
```

### MegatronEngine 初始化参数

```python
MegatronEngine(
    model_path: str,
    parallel_strategy: ParallelStrategy,
    dtype: torch.dtype,
    use_fp8: bool = False,              # 是否使用 FP8
    # ... 其他参数
)
```

### 推理引擎请求/响应

**请求**（`ModelRequest`）：

```python
{
    "rid": str,                         # 请求 ID
    "input_ids": List[int],             # 输入 token IDs
    "gconfig": GenerationHyperparameters, # 生成配置
    "tokenizer": PreTrainedTokenizerFast,
}
```

**响应**（`ModelResponse`）：

```python
{
    "input_tokens": List[int],          # 输入 tokens
    "output_tokens": List[int],         # 输出 tokens
    "logprobs": List[float],            # 对数概率
    "version": int,                     # 模型版本
}
```

## 测试与质量

- **测试覆盖**：
  - FSDP：`areal/tests/test_fsdp_*.py`（需 GPU）
  - Megatron：`areal/tests/test_megatron_*.py`（需多 GPU）
  - 推理引擎：`areal/tests/test_inference_engines.py`
- **质量工具**：Ruff、pre-commit hooks

## 常见问题 (FAQ)

### Q: 如何选择训练引擎？

A:
- **FSDP2**：适合单节点或小规模多节点训练，支持 LoRA
- **Megatron**：适合大规模多节点训练，支持 Pipeline Parallel、Expert Parallel

### Q: 如何选择推理引擎？

A:
- **SGLang**：支持 Data Parallel Attention、Expert Parallel，适合 MoE 模型
- **vLLM**：成熟稳定，支持 Tensor Parallel、Pipeline Parallel

### Q: 如何实现自定义引擎？

A: 继承 `TrainEngine` 或 `InferenceEngine`，实现所有抽象方法。参考 `fsdp_engine.py` 或 `sglang_remote.py`。

### Q: 如何处理 OOM？

A: 参考 `docs/best_practices/handling_oom.md`：
- 减少 batch size 或 micro-batch 数量
- 启用 gradient checkpointing
- 使用 FP8 或混合精度训练
- 增加 Tensor Parallel 或 Pipeline Parallel

## 相关文件清单

```
areal/engine/
├── __init__.py
├── fsdp_engine.py             # FSDP2 训练引擎（核心）
├── megatron_engine.py         # Megatron 训练引擎（核心）
├── sglang_remote.py           # SGLang 推理引擎（核心）
├── vllm_remote.py             # vLLM 推理引擎（核心）
├── core/
│   ├── __init__.py
│   └── train_engine.py        # TrainEngine 基类
├── ppo/
│   ├── actor.py               # PPO Actor
│   └── critic.py              # PPO Critic
├── sft/
│   └── lm_engine.py           # SFT 语言模型引擎
└── rw/
    └── rw_engine.py           # Reward Model 引擎
```

## 下一步建议

- 补充 Megatron 引擎的详细配置文档
- 添加推理引擎的性能对比测试
- 优化 FSDP 引擎的检查点保存/加载逻辑
