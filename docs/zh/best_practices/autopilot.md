# AReaL Autopilot（离线分析）

离线 Autopilot 分析器将 AReaL 训练配置和 PerfTracer 输出转换为阶段耗时汇总及后续诊断建议。它还可以把兼容的 `areal.tools.benchmark_fsdp` 结果作为调优候选附在报告中。

分析器只读配置和数据，不会修改配置、启动训练任务或部署资源。阶段耗时反映实际观测到的时间，不能单独证明 GPU、通信、存储或调度器是根因。

## 收集和分析

在训练配置中启用 `perf_tracer.enabled`，然后运行实验。PerfTracer 会为每个参与的 rank 写出一个 `traces-r{rank}.jsonl` 文件。将训练 YAML 和所有 trace 文件传给分析器：

```bash
python -m areal.tools.autopilot \
  --config examples/math/gsm8k_grpo.yaml \
  --traces /path/to/perf_tracer/traces-r*.jsonl \
  --output-dir /path/to/autopilot-report
```

输出目录必须不存在。目录中会生成供后续处理的 `report.json` 和方便审阅的 `report.md`。

## 添加 FSDP 测量结果

可选的 FSDP train-step benchmark 报告只有在 actor 使用 FSDP、分配的 world size 与节点拓扑匹配，并且配置模型路径与 benchmark 模型一致时才会纳入分析：

```bash
python -m areal.tools.autopilot \
  --config examples/math/gsm8k_grpo.yaml \
  --traces /path/to/perf_tracer/traces-r*.jsonl \
  --benchmark-report /path/to/fsdp-benchmark/report.json \
  --output-dir /path/to/autopilot-report
```

该 benchmark 使用合成 SFT token，而不是目标 RL 工作负载。报告中的配置只是后续 RL 实验的候选依据，不会自动应用，也不是通用调优结论。

## 阅读报告

分析器识别的 `train.*` 完整事件会归入 rollout、data、training、weight-update、checkpoint、memory-transfer、evaluation 和 instrumentation 阶段。每个操作先在同一 role 内取最慢 rank；顺序操作在各 role 内累加，阶段耗时再取最慢 role。这样不会把并行 rank 或 role 的时间相加。阶段占比只按输入 trace 中存在的已识别阶段计算。报告会整体及按 role 列出未观测到的已知 span，便于发现某个角色缺少埋点。

每条建议都链接到触发它的具体 span，并说明下一步要采集什么数据。报告会列出缺少的埋点，避免把未观测到的阶段误认为耗时为零。要判断阶段耗时对应哪类资源，应使用现有 profiling 工具采集代表性 step 的 CPU/GPU 活动。

如果提供 benchmark 报告，候选行会包含实测吞吐、峰值已分配显存、benchmark 工作负载、dtype、attention 实现、优化器、设备和运行环境信息。只有 benchmark 与训练配置兼容时，才会把配置差异列为待验证候选。
