# Cross-Benchmark 实验运行指南

## 概述

Cross-benchmark实验用于测试在一个benchmark上训练的memory是否可以迁移到其他benchmark。

## 文件说明

- `run_cross_benchmark_experiment.py` - 主Python脚本，支持完整的跨benchmark实验
- `run_bcb_experiment.sh` - BCB单独实验的简化脚本
- `cross_benchmark_experiment.sh` - 完整的shell脚本版本
- `quick_start_cross_benchmark.sh` - 使用srun快速启动

## 环境准备

### 1. 设置API密钥

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://your-api-base-url/v1"  # 可选
export LLM_MODEL="gpt-4o-mini"  # 可选
```

### 2. 数据准备

#### BigCodeBench (BCB)
数据已在 `data/bigcodebench/bigcodebench_full.jsonl`

#### LLB (Lifelong Benchmark)
数据已在 `data/llb/` 目录下

#### HLE (Humanity's Last Exam)
**需要手动下载**（当前网络无法访问HuggingFace）

```python
# 在有网络的环境执行:
from datasets import load_dataset
ds = load_dataset('cais/hle', split='test')
ds.to_parquet('data/hle/hle_test.parquet')
```

或者手动从 https://huggingface.co/datasets/cais/hle 下载parquet文件到 `data/hle/hle_test.parquet`

## 运行实验

### 方式1: 直接运行BCB实验

```bash
cd /storage/openpsi/users/yl/agent-memory/MemRL
export LLM_API_KEY="your-key"
bash scripts/run_bcb_experiment.sh
```

### 方式2: 使用Python脚本（推荐）

```bash
cd /storage/openpsi/users/yl/agent-memory/MemRL

# 在BCB上训练，测试LLB
python scripts/run_cross_benchmark_experiment.py \
    --source bcb \
    --targets llb \
    --api_key "your-key" \
    --mode local

# 在LLB上训练，测试HLE和BCB
python scripts/run_cross_benchmark_experiment.py \
    --source llb \
    --targets hle bcb \
    --api_key "your-key" \
    --mode local
```

### 方式3: 使用srun在计算节点运行

```bash
export LLM_API_KEY="your-key"
export SOURCE_BENCHMARK=bcb
export TARGET_BENCHMARKS="llb"

bash scripts/quick_start_cross_benchmark.sh bcb llb
```

### 方式4: 使用Singularity容器（如果需要）

```bash
# 注意: 当前环境singularity可能有权限问题
singularity exec /storage/openpsi/images/areal-latest.sif \
    python scripts/run_cross_benchmark_experiment.py \
    --source bcb --targets llb --api_key "your-key"
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --source | 训练benchmark (bcb/llb/hle/alf) | 必填 |
| --targets | 测试benchmark列表 | 必填 |
| --api_key | LLM API密钥 | 必填 |
| --base_url | API base URL | https://api.openai.com/v1 |
| --model | LLM模型名称 | gpt-4o-mini |
| --epochs | 训练轮数 | 10 |
| --batch_size | 批处理大小 | 5 |
| --mode | 运行模式 (local/srun) | local |

## 输出结果

结果保存在 `results/` 目录：
- `results/bigcodebench_eval/` - BCB评估结果
- `results/llb/` - LLB评估结果  
- `results/hle/` - HLE评估结果
- `results/cross_benchmark_report_*.json` - 跨benchmark分析报告

## 故障排除

1. **HuggingFace连接超时**: 当前环境网络无法访问，需要手动下载数据
2. **Singularity权限错误**: 使用Python直接运行替代
3. **memrl未安装**: `pip install -e .`
