# Cross-Benchmark Memory Transfer Experiment Scripts

## 概述

这套脚本用于在一个benchmark上训练MemRL的memory，然后将其迁移到其他benchmark进行测试，以验证memory的泛化能力。

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_cross_benchmark_experiment.py` | **推荐使用** - Python版本的实验脚本，功能最完整 |
| `quick_start_cross_benchmark.sh` | 快速启动脚本，使用srun提交到计算节点 |
| `cross_benchmark_experiment.sh` | 完整的bash实验脚本，支持local/srun/sbatch模式 |
| `analyze_cross_benchmark_results.py` | 结果分析脚本 |
| `generate_eval_config.py` | 配置文件生成辅助脚本 |
| `run_single_eval.sh` | 单个评估任务脚本 (用于SLURM依赖调度) |

## 使用方法

### 方法1: Python脚本 (推荐)

```bash
# 基本用法: 在LLB上训练，测试HLE和BCB
python scripts/run_cross_benchmark_experiment.py \
    --source llb \
    --targets hle bcb \
    --api_key YOUR_API_KEY

# 使用srun在计算节点运行
python scripts/run_cross_benchmark_experiment.py \
    --source llb \
    --targets hle bcb \
    --api_key YOUR_API_KEY \
    --mode srun

# 完整参数示例
python scripts/run_cross_benchmark_experiment.py \
    --source llb \
    --targets hle bcb alf \
    --api_key YOUR_API_KEY \
    --base_url https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --epochs 10 \
    --batch_size 5 \
    --llb_task os \
    --mode srun \
    --partition all \
    --gpus 1
```

### 方法2: 快速启动脚本

```bash
# 设置环境变量
export LLM_API_KEY=your-api-key
export LLM_BASE_URL=https://api.openai.com/v1  # 可选
export LLM_MODEL=gpt-4o-mini                    # 可选

# 运行实验
./scripts/quick_start_cross_benchmark.sh llb hle bcb
```

### 方法3: 完整bash脚本

```bash
# 设置环境变量
export LLM_API_KEY=your-api-key
export SOURCE_BENCHMARK=llb
export TARGET_BENCHMARKS="hle bcb"

# 本地运行
bash scripts/cross_benchmark_experiment.sh local

# srun模式
bash scripts/cross_benchmark_experiment.sh srun

# sbatch模式 (提交作业)
bash scripts/cross_benchmark_experiment.sh sbatch
```

## 支持的Benchmarks

| Benchmark | 代码 | 说明 |
|-----------|------|------|
| LifelongAgentBench | `llb` | 包含os和db两个任务 |
| BigCodeBench | `bcb` | 代码生成任务 |
| HLE (Humanity's Last Exam) | `hle` | 高难度问答 |
| ALFWorld | `alf` | 家庭环境任务 |

## 配置参数

### API配置
- `--api_key` / `LLM_API_KEY`: API密钥 (必须)
- `--base_url` / `LLM_BASE_URL`: API地址
- `--model` / `LLM_MODEL`: 模型名称

### 训练配置
- `--epochs` / `NUM_SECTIONS`: 训练轮数 (默认: 10)
- `--batch_size` / `BATCH_SIZE`: 批次大小 (默认: 5)
- `--seed`: 随机种子 (默认: 42)
- `--llb_task`: LLB任务类型 (os/db)

### SLURM配置
- `--partition`: 计算分区 (默认: all)
- `--gpus`: GPU数量 (默认: 1)
- `--cpus`: CPU数量 (默认: 8)
- `--mem`: 内存 (默认: 32G)
- `--time`: 时间限制 (默认: 24:00:00)

## 输出结果

实验结果保存在 `results/` 目录下:

```
results/
├── llb/
│   └── exp_cross_llb_20240417_123456_train/
│       ├── snapshot/
│       │   └── 10/  # checkpoint
│       ├── local_cache/
│       └── results.json
├── hle/
│   └── exp_cross_llb_20240417_123456_hle_eval/
│       └── results.json
├── bcb/
│   └── ...
└── cross_benchmark_report_cross_llb_20240417_123456.json  # 汇总报告
```

## 结果分析

实验完成后会自动生成分析报告。也可以手动运行:

```bash
python scripts/analyze_cross_benchmark_results.py \
    --results_dir ./results \
    --experiment_name cross_llb_20240417_123456 \
    --source_benchmark llb \
    --target_benchmarks hle bcb
```

报告内容包括:
- 源benchmark训练结果 (成功率、memory数量、Q值等)
- 各目标benchmark评估结果
- Memory迁移增益分析
- 实验摘要

## 注意事项

1. **API密钥**: 请确保设置了正确的API密钥
2. **数据文件**: LLB需要 `data/llb/` 下的数据文件
3. **依赖安装**: 确保已安装memrl模块 (`pip install -e .`)
4. **计算资源**: BCB评估需要较多内存和时间
5. **Checkpoint**: 训练会自动保存checkpoint到 `snapshot/` 目录

## 示例: 完整实验流程

```bash
cd /storage/openpsi/users/yl/agent-memory/MemRL

# 1. 设置API密钥
export LLM_API_KEY=sk-xxxxx

# 2. 在LLB-OS上训练，测试HLE
python scripts/run_cross_benchmark_experiment.py \
    --source llb \
    --targets hle \
    --api_key $LLM_API_KEY \
    --epochs 10 \
    --mode srun

# 3. 查看结果
cat results/cross_benchmark_report_*.json | python -m json.tool
```
