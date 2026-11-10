# Elastic 快速命令

在 AReaL 仓库根目录启用原生 Elastic，先填好模型、GPU 池、`state_root` 和 `checkpoint_estimated_bytes`：

```bash
CUDA_VISIBLE_DEVICES="" \
python -m examples.v2_online_workflows.train \
  --config examples/v2_online_workflows/configs/dapo.yaml \
  elastic.enabled=true
```

当前原生入口使用 Megatron＋SGLang 固定布局，保留完整 checkpoint 与观测；Megatron 动态布局切换待实现。

CPU 实验在工作区根目录运行：

```bash
python local/simulation/simulate_online.py \
  --cost-models analytical \
  --config local/simulation/configs/smoke.json \
  --search-workers 2 --decision-clock fixed
```

输出自动保存在 `local/simulation/outputs/<配置名-时间戳>/`。

生成 workload 和单次搜索修订演示见[本地实验命令](../../../local/simulation/command.md)。
