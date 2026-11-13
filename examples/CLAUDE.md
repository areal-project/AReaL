[根目录](../CLAUDE.md) > **examples**

# examples - 训练脚本与配置

## 变更记录 (Changelog)

### 2026-01-31 - 初始化

- 模块文档创建
- 识别 10+ 个示例场景

---

## 模块职责

提供各种场景的训练脚本与配置文件：

- **数学推理**：GSM8K、BOBA（GRPO、PPO、SFT、DAPO、LitePPO、RLOO 等）
- **多轮数学**：多轮对话数学推理
- **视觉语言模型**：CLEVR 计数、Geometry3K
- **搜索 Agent**：端到端推理、搜索、浏览、总结
- **工具集成推理**：TIR（Tool-Integrated Reasoning）
- **RLHF**：奖励模型训练
- **LoRA**：低秩适应训练
- **实验性**：代理模式、近似优化

## 入口与启动

### 数学推理（GSM8K）

```bash
# GRPO 训练
python3 -m areal.launcher.local \
  examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml

# SFT 训练
python3 -m areal.launcher.local \
  examples/math/gsm8k_sft.py \
  --config examples/math/gsm8k_sft.yaml

# 评估
python3 examples/math/gsm8k_eval.py \
  --model_path /path/to/checkpoint
```

### 多轮数学推理

```bash
python3 -m areal.launcher.local \
  examples/multi_turn_math/gsm8k_rl_mt.py \
  --config examples/multi_turn_math/gsm8k_grpo_mt.yaml
```

### 视觉语言模型

```bash
# CLEVR 计数
python3 -m areal.launcher.local \
  examples/vlm/clevr_count_70k_grpo.py \
  --config examples/vlm/clevr_count_70k_grpo.yaml

# Geometry3K
python3 -m areal.launcher.local \
  examples/vlm/geometry3k_grpo.py \
  --config examples/vlm/geometry3k_grpo.yaml
```

## 对外接口

### 训练脚本

| 脚本                          | 路径                              | 职责                                   |
| ----------------------------- | --------------------------------- | -------------------------------------- |
| `gsm8k_rl.py`                 | `math/gsm8k_rl.py`                | GSM8K RL 训练（GRPO/PPO/RLOO 等）      |
| `gsm8k_sft.py`                | `math/gsm8k_sft.py`               | GSM8K SFT 训练                         |
| `gsm8k_eval.py`               | `math/gsm8k_eval.py`              | GSM8K 评估                             |
| `gsm8k_rl_mt.py`              | `multi_turn_math/gsm8k_rl_mt.py`  | 多轮数学推理训练                       |
| `clevr_count_70k_grpo.py`     | `vlm/clevr_count_70k_grpo.py`     | CLEVR 计数 GRPO 训练                   |
| `geometry3k_grpo.py`          | `vlm/geometry3k_grpo.py`          | Geometry3K GRPO 训练                   |
| `train_tir.py`                | `tir/train_tir.py`                | TIR 训练                               |
| `train_agents.py`             | `openai_agents/train_agents.py`   | OpenAI Agents 训练                     |

### 配置文件

| 配置文件                      | 路径                              | 职责                                   |
| ----------------------------- | --------------------------------- | -------------------------------------- |
| `gsm8k_grpo.yaml`             | `math/gsm8k_grpo.yaml`            | GSM8K GRPO 配置                        |
| `gsm8k_ppo.yaml`              | `math/gsm8k_ppo.yaml`             | GSM8K PPO 配置                         |
| `gsm8k_sft.yaml`              | `math/gsm8k_sft.yaml`             | GSM8K SFT 配置                         |
| `gsm8k_dapo_dynamic_bs.yaml`  | `math/gsm8k_dapo_dynamic_bs.yaml` | GSM8K DAPO 动态批次配置                |
| `gsm8k_grpo_lora.yaml`        | `lora/gsm8k_grpo_lora.yaml`       | GSM8K GRPO LoRA 配置                   |
| `gsm8k_grpo_megatron.yaml`    | `math/gsm8k_grpo_megatron.yaml`   | GSM8K GRPO Megatron 配置               |

## 关键依赖与配置

### 外部依赖

- `torch`：训练框架
- `transformers`：模型加载
- `datasets`：数据集加载
- `wandb`：实验跟踪（可选）

### 内部依赖

- `areal.api.cli_args`：配置数据类
- `areal.workflow`：工作流实现
- `areal.reward`：奖励函数
- `areal.dataset`：数据集加载器
- `areal.launcher`：启动器

## 数据模型

### 配置文件结构（YAML）

```yaml
# 模型配置
model:
  path: "Qwen/Qwen2-1.5B-Instruct"
  dtype: "bfloat16"

# 训练配置
training:
  algorithm: "grpo"
  n_epochs: 1
  batch_size: 128
  learning_rate: 1e-5

# 生成配置
generation:
  temperature: 0.7
  top_p: 0.9
  max_tokens: 512

# 并行配置
parallel:
  data_parallel_size: 2
  tensor_parallel_size: 1

# 工作流配置
workflow:
  type: "rlvr"
  n_samples: 4

# 数据集配置
dataset:
  name: "gsm8k"
  split: "train"
```

## 测试与质量

- **测试覆盖**：`areal/tests/test_examples.py`（验证示例脚本可运行）
- **质量工具**：Ruff、pre-commit hooks

## 常见问题 (FAQ)

### Q: 如何修改训练超参数？

A: 编辑对应的 YAML 配置文件，或通过命令行覆盖：

```bash
python3 -m areal.launcher.local \
  examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  training.learning_rate=5e-6 \
  training.batch_size=64
```

### Q: 如何在多节点上运行？

A: 使用 Ray 或 Slurm 启动器：

```bash
# Ray
python3 -m areal.launcher.ray \
  examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=8

# Slurm
python3 -m areal.launcher.slurm \
  examples/math/gsm8k_rl.py \
  --config examples/math/gsm8k_grpo.yaml \
  --partition=gpu \
  --nodes=2 \
  --gpus-per-node=8
```

### Q: 如何添加新的示例？

A: 参考现有示例（如 `math/gsm8k_rl.py`）：
1. 创建训练脚本（定义 workflow、reward、dataset）
2. 创建配置文件（YAML）
3. 添加 README.md 说明

## 相关文件清单

```
examples/
├── math/                      # 数学推理
│   ├── gsm8k_rl.py            # RL 训练脚本（核心）
│   ├── gsm8k_sft.py           # SFT 训练脚本
│   ├── gsm8k_eval.py          # 评估脚本
│   ├── gsm8k_grpo.yaml        # GRPO 配置（核心）
│   ├── gsm8k_ppo.yaml         # PPO 配置
│   ├── gsm8k_sft.yaml         # SFT 配置
│   └── README.md
├── multi_turn_math/           # 多轮数学推理
│   ├── gsm8k_rl_mt.py
│   ├── gsm8k_grpo_mt.yaml
│   └── README.md
├── vlm/                       # 视觉语言模型
│   ├── clevr_count_70k_grpo.py
│   ├── geometry3k_grpo.py
│   └── *.yaml
├── lora/                      # LoRA 训练
│   ├── gsm8k_grpo_lora.py
│   └── gsm8k_grpo_lora.yaml
├── tir/                       # 工具集成推理
│   ├── train_tir.py
│   ├── tir_workflow.py
│   └── tools/
├── search_agent/              # 搜索 Agent
│   └── tongyi_deepresearch/
├── openai_agents/             # OpenAI Agents
│   └── train_agents.py
├── alignment/                 # RLHF
│   └── hhrlhf_rw.py
├── experimental/              # 实验性功能
│   ├── proxy/
│   └── prox_approx/
└── skypilot/                  # SkyPilot 部署
    └── *.sky.yaml
```

## 下一步建议

- 补充各示例的详细文档与性能基准
- 添加更多算法的示例（GSPO、Dr.GRPO 等）
- 优化配置文件的默认参数
