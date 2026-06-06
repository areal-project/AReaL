# 快速使用指南

## 1. 启动服务

### 单节点启动（使用 srun）

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
         bash code_verify_service/launch_service.sh 8000
```

### 多节点启动

```bash
bash scripts/code_verify_service/launch_multiple_services.sh 8 12000 path-to-results/llm_test_time_scaling/data/local_data/lcb_testcases/data tolerence_verify_services.txt
```

这将启动 4 个服务，端口从 8000 开始。

## 2. 使用 RemoteLCBProEvaluator

```python
import asyncio
from src.evaluation.remote_lcb_pro_evaluator import RemoteLCBProEvaluator

async def main():
    # 创建远程评估器
    evaluator = RemoteLCBProEvaluator(
        service_url="http://<HOST_IP>:8000",  # 替换为实际的服务地址
        timeout=300,
    )
    
    # 评估代码
    result = await evaluator.evaluate(
        problem="Problem statement",
        solution="#include <iostream>\nint main() { return 0; }",
        problem_id="1983A",
        language="cpp",
    )
    
    print(f"结果: {result.is_correct}, 得分: {result.score}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 3. 测试服务

```bash
# 健康检查
curl http://<HOST_IP>:8000/health

# 评测代码
curl -X POST http://<HOST_IP>:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "code": "#include <iostream>\nint main() { return 0; }",
    "problem_id": "1983A"
  }'
```

## 注意事项

1. 确保测试数据目录存在：`path-to-results/LLM_Test-time_Scaling/data/local_data/lcb_testcases/data`
2. 服务启动后，会在控制台输出服务地址和端口
3. 多节点部署时，每个服务使用不同的端口
4. 可以通过服务注册表文件查看所有运行的服务
