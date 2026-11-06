# Elastic 示例

在 AReaL 仓库根目录运行。`basic` 使用 Basic v1，`reference` 为默认版本；模拟和修订使用已冻结的组件与输入。

## Online 模拟

先生成工作负载：

```bash
python -m areal.elastic.simulation.workload \
  --spec examples/autopilot/elastic/workload-spec.json \
  --output "$WORKLOAD_DIR"
```

可用 `workload-regimes/*.json` 替换 `--spec`，或用 `--arrivals workload-arrivals/*.json`
覆盖到达模式。输出包含 `spec.json`、`workload.jsonl` 和 `manifest.json`；JSONL 每行一个任务，保存到达时间、调用依赖和
token／工具工作量。

准备冻结校准包、mechanism fits 和 publication metadata，再运行：

```bash
python examples/autopilot/elastic/simulate_online.py \
  --reconfigurator reference --workload "$WORKLOAD_DIR/workload.jsonl" \
  --config examples/autopilot/elastic/online-v3-example.json \
  --bundle "$CALIBRATION_BUNDLE" --mechanism-fits "$MECHANISM_FITS" \
  --publication-metadata "$PUBLICATION_METADATA" --output "$OUTPUT"
```

- `OUTPUT` 使用新目录；默认同时运行 fixed／dynamic，`--mode` 可单选。
- `--search-workers 4` 启用候选进程池，`1` 为串行；默认计入搜索／复核的 CPU 墙钟，`--decision-clock fixed`
  用于确定性对照。
- Basic 使用 `--reconfigurator basic` 与 `online-v1-example.json`。
- 配置中 `handoff.mode` 选择 `hot`／`cold`；Cold 需提供切换阶段耗时。结果保存输入身份、决策、事件、请求、训练顺序与 GPU 积分。

## CPU 修订示例

```bash
python -m examples.autopilot.revising --demo --target evaluator --output /tmp/online-evaluator-revision
python -m examples.autopilot.revising --demo --target optimizer --output /tmp/online-optimizer-revision
```

每次训练使用独立输出目录。多个修订 Cycle 共用本次训练的历史时，传入
`--history-root /tmp/training-001/revision --output /tmp/training-001/revision/cycle-001/optimizer`。
`history.json` 保存各轮摘要，下一轮 Observation 自动带入同一修订对象的历史。

Evaluator 当前修订其内部 CostModels，外层 Revision Benchmark 评测代码修改。 非 demo 还需
`--profile`、`--cases`，以及 `--agent-command` 或 `--llm-model`。profile／cases 的 factory
契约见外层[Elastic Revision](../../../../docs/implementation/elastic-revision.md)。

### DeepSeek API

在运行终端设置 `DEEPSEEK_API_KEY` 后，调用现有 Model API 入口。

```bash
python -m examples.autopilot.revising --demo --target optimizer \
  --output /tmp/deepseek-training-001/revision/cycle-001/optimizer \
  --history-root /tmp/deepseek-training-001/revision --max-rounds 2 \
  --llm-model deepseek-flash --llm-base-url https://api.deepseek.com \
  --llm-api-key-env DEEPSEEK_API_KEY --llm-response-format json_object \
  --llm-token-parameter max_tokens --llm-reasoning-effort none
```

此命令用公式场景验证提案与评测链路。实际搜索案例使用 `--profile` 和 `--cases` 替换 `--demo`。 参数依据
[DeepSeek 接口](https://api-docs.deepseek.com/)与
[JSON Output](https://api-docs.deepseek.com/guides/json_mode/)。

## Workload 变化触发 Revision

```bash
python -m examples.autopilot.elastic.simulate_revision --demo --output /tmp/elastic-revision-demo
```

`revision-example.json` 配置检测窗口、阈值、评测范围和修订预算。此演示使用显式 CPU 公式、重复查询的初态 Optimizer 和本地受控
proposer。结果及每轮材料写入输出目录。Revision 同步执行，墙钟耗时另行记录。实际场景使用 `--profile`、`--workload`，以及
`--model-settings` 或 `--agent-command`。

## 原生训练

使用[任务工作流](../../task_workflows/README.md)或[Hermes 示例](../../hermes/README.md)启动原生
`PPOTrainer`。客户端使用 `resident` 供数；手动和自动调整都进入 `PPOTrainer.reconfigure()`。原生自动控制通过
`--elastic-profile`／`--elastic-settings` 配置 `ReconfiguratorAdapter`，在等待 batch
及更新完成后轮询。profile 的 `slo` 使用 `ttft_seconds: 5`、`tpot_seconds: 0.05`。

原生预测需提供匹配后端的组件模型、真实 batch／staleness 及 Cold 切换阶段耗时。首末 token 时间缺测保持未知，完整请求观测可通过 adapter 的
`observation_provider` 提供。Revision 接入暂缓。

完整实现、实验结果与机器操作说明维护在外层[项目文档](../../../../docs/README.md)。
