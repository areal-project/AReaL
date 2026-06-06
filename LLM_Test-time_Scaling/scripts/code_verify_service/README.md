# Code Verify Service

远程 C++ 代码评测服务，用于在 SLURM 集群上运行代码验证，避免本地运行的性能瓶颈。

## 概述

该服务提供了一个 HTTP API 接口，用于评测 C++ 代码。服务运行在 SLURM 集群的 CPU 节点上（32 CPU，0 GPU），可以通过 HTTP 请求进行代码评测。

## 文件结构

```
code_verify_service/
├── __init__.py              # 包初始化文件
├── server.py                # HTTP API 服务器
├── launch_service.sh        # 单节点启动脚本
├── launch_multiple_services.sh  # 多节点启动脚本
└── README.md               # 本文件
```

## 安装依赖

```bash
pip install fastapi uvicorn aiohttp pydantic
```

或安装完整依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 启动单个服务

使用 `srun` 启动单个服务：

```bash
srun --job-name=tts_cpu_eval \
     --mpi=pmi2 \
     --chdir=path-to-results/LLM_Test-time_Scaling \
     --ntasks=1 \
     --cpus-per-task=64 \
     --mem-per-cpu=10000M \
     --gres=gpu:0 \
     singularity exec --pid --env=ENV_VAR=env_var --nv \
         --bind /storage:/storage \
         /storage/openpsi/images/areal-v0.3.3-sglang-v0.5.2-vllm-v0.10.2-v3.sif \
         bash code_verify_service/launch_service.sh <PORT> <DATA_DIR> [SERVICE_REGISTRY]
```

参数：
- `PORT`: 服务端口（默认：8000）
- `DATA_DIR`: 测试数据目录（默认：`path-to-results/LLM_Test-time_Scaling/data/local_data/lcb_testcases/data`）
- `SERVICE_REGISTRY`: 服务注册表文件路径（可选）

### 2. 启动多个服务

使用批量启动脚本：

```bash
bash code_verify_service/launch_multiple_services.sh <NUM_NODES> <START_PORT> [DATA_DIR] [SERVICE_REGISTRY]
```

示例：

```bash
bash scripts/code_verify_service/launch_multiple_services.sh 16 6000
```

这将启动 4 个服务，端口从 8000 开始（8000, 8001, 8002, 8003）。

### 3. 检查服务健康状态

```bash
curl http://<HOST_IP>:<PORT>/health
```

### 4. 使用 RemoteLCBProEvaluator

在代码中使用远程评估器：

```python
from src.evaluation.remote_lcb_pro_evaluator import RemoteLCBProEvaluator

# 创建远程评估器
evaluator = RemoteLCBProEvaluator(
    service_url="http://<HOST_IP>:<PORT>",
    data_dir="/path/to/data",  # 可选，服务端会使用默认路径
    timeout=300,
)

# 评估代码
result = await evaluator.evaluate(
    problem="Problem statement",
    solution="#include <iostream>\nint main() { return 0; }",
    problem_id="1983A",
    language="cpp",
)

print(f"Passed: {result.is_correct}, Score: {result.score}")
```

## API 接口

### POST /verify

评测 C++ 代码。

**请求体：**

```json
{
    "code": "#include <iostream>\nint main() { return 0; }",
    "problem_id": "1983A",
    "data_dir": "/path/to/data",  // 可选
    "compile_timeout": 30          // 可选，默认 30 秒
}
```

**响应：**

```json
{
    "success": true,
    "results": [true, true, false],
    "passed": 2,
    "total": 3,
    "is_correct": false,
    "score": 0.6666666666666666,
    "feedback": "Passed 2/3 test cases",
    "metadata": {
        "error_message": "...",
        "test_0_...": "...",
        ...
    }
}
```

### GET /health

健康检查接口。

**响应：**

```json
{
    "status": "healthy",
    "service": "code_verify"
}
```

## 服务注册表

服务启动后，会在注册表文件中记录服务信息，格式：

```
JOB_ID|HOSTNAME|HOST_IP|PORT|DATA_DIR|START_TIME|STATUS|END_TIME|EXIT_CODE
```

可以使用文本编辑器查看服务状态，或使用脚本解析。

## 注意事项

1. 确保测试数据目录存在且可访问
2. 服务需要能够访问测试数据目录和 testlib.h
3. 建议在启动服务前检查端口是否被占用
4. 多节点部署时，确保每个节点使用不同的端口
5. 服务会在后台运行，可以使用 `kill` 命令停止

## 故障排除

1. **服务无法启动**：检查端口是否被占用，数据目录是否存在
2. **评测失败**：检查问题 ID 是否正确，数据目录是否包含对应的测试用例
3. **连接超时**：增加 `timeout` 参数，或检查网络连接
4. **编译错误**：检查代码是否有效，编译器是否可用
