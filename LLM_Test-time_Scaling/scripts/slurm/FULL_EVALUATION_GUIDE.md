# 完整评测流程指南

本指南介绍如何在 SLURM 集群上启动 LLM 服务并运行 IMOBench 和 LiveCodeBench-Pro 的完整评测流程。

## 快速开始

### 方法 1: 使用完整流程脚本（推荐）

```bash
# 在 SLURM 节点上提交完整评测任务
sbatch scripts/slurm/run_full_evaluation.sh \
    gpt-oss-120b \
    1 \
    all

# 或者使用 Python 脚本（可以在登录节点运行）
python scripts/run_full_evaluation.py \
    --model-path gpt-oss-120b \
    --num-services 4 \
    --benchmarks all
```

### 方法 2: 分步执行

#### 步骤 1: 启动 LLM 服务

```bash
# 启动 4 个服务实例
bash scripts/slurm/launch_multiple_services.sh \
    1 \
    gpt-oss-120b \
    8000
```

#### 步骤 2: 等待服务就绪

```bash
# 等待服务启动并健康检查
python scripts/slurm/wait_for_services.py \
    --timeout 600 \
    --min-healthy 1
```

#### 步骤 3: 运行评测

```bash
# 导出服务 URL 环境变量
python scripts/slurm/service_manager.py export -o env_vars.sh
source env_vars.sh

# Evaluate from rollouts
python -m scripts.evaluate_imobench \
    --rollout-file results/imobench_rollouts/baseline_imobench_20240115_103000.json

# Evaluate by calling model directly (using existing LLM service)
export OPENAI_API_BASE="xxx"
python -m scripts.evaluate_imobench \
    --model-name openai/gpt-oss-120b \
    --num-samples 1 \
    --output-dir results/imobench_evaluation

# Or use SGLANG_API_BASES (supports multiple comma-separated URLs)
export SGLANG_API_BASES="xxx,xxx"
python -m scripts.evaluate_imobench \
    --model-name openai/gpt-oss-120b \
    --num-samples 1

# 运行 LiveCodeBench-Pro 评测
python scripts/run_lcb_pro_experiment.py
```

#### 步骤 4: 清理服务（可选）

```bash
# 取消所有运行中的服务
python scripts/slurm/service_manager.py cancel --force
```

## 详细说明

### 1. 完整流程脚本 (`run_full_evaluation.py`)

这是主要的脚本，自动执行所有步骤：

```bash
python scripts/run_full_evaluation.py \
    --model-path gpt-oss-120b \
    --model-name openai/gpt-oss-120b \
    --num-services 1 \
    --start-port 8000 \
    --benchmarks imobench \
    --output-dir results \
    --timeout 600 \
    --cleanup
```

**参数说明：**

- `--model-path`: 模型路径（必需）
- `--model-name`: API 使用的模型名称（可选，默认从路径推导）
- `--num-services`: 启动的服务实例数量（默认：4）
- `--start-port`: 起始端口号（默认：8000）
- `--benchmarks`: 要运行的评测（`imobench`, `lcb_pro`, `all`，默认：`all`）
- `--output-dir`: 结果输出目录（默认：`results`）
- `--timeout`: 等待服务就绪的超时时间（秒，默认：600）
- `--skip-launch`: 跳过启动服务（使用已有服务）
- `--skip-wait`: 跳过等待服务（假设服务已就绪）
- `--cleanup`: 评测完成后清理服务

### 2. SLURM 批处理脚本 (`run_full_evaluation.sh`)

用于在 SLURM 集群上提交完整评测任务：

```bash
sbatch scripts/slurm/run_full_evaluation.sh \
    <MODEL_PATH> \
    <NUM_SERVICES> \
    [BENCHMARKS] \
    [START_PORT] \
    [OUTPUT_DIR]
```

**示例：**

```bash
# 运行所有评测
sbatch scripts/slurm/run_full_evaluation.sh \
    gpt-oss-120b \
    4 \
    all

# 只运行 IMOBench
sbatch scripts/slurm/run_full_evaluation.sh \
    gpt-oss-120b \
    4 \
    imobench

# 自定义端口和输出目录
sbatch scripts/slurm/run_full_evaluation.sh \
    gpt-oss-120b \
    4 \
    all \
    9000 \
    results/my_experiment
```

### 3. 等待服务脚本 (`wait_for_services.py`)

等待服务启动并健康检查：

```bash
python scripts/slurm/wait_for_services.py \
    --registry services.txt \
    --timeout 600 \
    --check-interval 10 \
    --min-healthy 1
```

**参数说明：**

- `--registry`: 服务注册表文件（默认：`services.txt`）
- `--timeout`: 最大等待时间（秒，默认：600）
- `--check-interval`: 健康检查间隔（秒，默认：10）
- `--min-healthy`: 所需的最少健康服务数（默认：1）

### 4. 评测脚本

#### IMOBench (`run_imobench_experiment.py`)

运行 IMOBench 数学问题评测：

```bash
# 需要先设置环境变量
export OPENAI_API_BASE="xxx,..."
export OPENAI_API_KEY="None"  # 如果使用本地服务

python scripts/run_imobench_experiment.py
```

#### LiveCodeBench-Pro (`run_lcb_pro_experiment.py`)

运行 LiveCodeBench-Pro 编程问题评测：

```bash
# 需要先设置环境变量
export OPENAI_API_BASE="xxx,..."
export OPENAI_API_KEY="None"

python scripts/run_lcb_pro_experiment.py
```

## 工作流程

### 完整流程

1. **启动服务**：在多个 SLURM 节点上启动 LLM 服务实例
2. **服务注册**：每个服务自动注册到 `services.txt`
3. **等待就绪**：检查服务健康状态，等待所有服务就绪
4. **运行评测**：
   - 加载服务 URL
   - 运行 IMOBench 评测
   - 运行 LiveCodeBench-Pro 评测
5. **保存结果**：结果保存到 `results/` 目录
6. **清理服务**（可选）：取消所有服务作业

### 服务管理

查看运行中的服务：

```bash
python scripts/slurm/service_manager.py list
```

检查服务健康状态：

```bash
python scripts/slurm/service_manager.py health
```

获取服务 URL：

```bash
python scripts/slurm/service_manager.py urls
```

导出环境变量：

```bash
python scripts/slurm/service_manager.py export -o env_vars.sh
source env_vars.sh
```

## 输出结果

评测结果保存在 `results/` 目录下，格式为：

```
results/
├── imobench_rollouts/
│   ├── baseline_imobench_20240115_103000.json
│   └── self_eval_sequential_imobench_20240115_104500.json
└── lcb_pro_rollouts/
    ├── baseline_lcb_pro_20240115_110000.json
    └── self_eval_sequential_lcb_pro_20240115_111500.json
```

每个结果文件包含：
- 实验配置
- 每个问题的详细结果
- Pass@1 和 Pass@k 指标
- Token 使用统计
- 时间信息

## 故障排除

### 服务启动失败

```bash
# 检查 SLURM 作业日志
tail -f logs/sglang_<JOB_ID>_<NODE>.out
tail -f logs/sglang_<JOB_ID>_<NODE>.err

# 检查节点 GPU
srun --gres=gpu:8 nvidia-smi
```

### 服务未就绪

```bash
# 手动检查服务健康
python scripts/slurm/service_manager.py health

# 检查服务注册表
cat services.txt

# 测试服务端点
curl http://<HOST_IP>:<PORT>/health
```

### 评测失败

```bash
# 检查环境变量
echo $OPENAI_API_BASE
echo $OPENAI_API_KEY

# 测试 API 连接
python -c "import requests; print(requests.get('$OPENAI_API_BASE/health').text)"
```

## 最佳实践

1. **从小规模开始**：先用 2-4 个服务测试，确认流程正常后再扩展
2. **监控资源**：使用 `squeue` 和 `nvidia-smi` 监控资源使用
3. **保存日志**：所有输出和错误日志保存在 `logs/` 目录
4. **定期清理**：定期清理旧的服务注册表和日志文件
5. **使用唯一端口**：避免端口冲突，使用不同的起始端口

## 示例：完整运行

```bash
# 1. 提交完整评测任务
sbatch scripts/slurm/run_full_evaluation.sh \
    gpt-oss-120b \
    4 \
    all

# 2. 监控作业
squeue -u $USER

# 3. 查看日志
tail -f logs/full_eval_<JOB_ID>.out

# 4. 查看结果
ls -lh results/*/
```

## 相关文件

- `scripts/run_full_evaluation.py`: 主流程脚本
- `scripts/slurm/run_full_evaluation.sh`: SLURM 批处理脚本
- `scripts/slurm/wait_for_services.py`: 服务等待脚本
- `scripts/run_imobench_experiment.py`: IMOBench 评测脚本
- `scripts/run_lcb_pro_experiment.py`: LiveCodeBench-Pro 评测脚本
- `scripts/slurm/service_manager.py`: 服务管理工具
- `scripts/slurm/launch_multiple_services.sh`: 批量启动服务脚本

