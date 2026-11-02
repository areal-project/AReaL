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
python -m examples.autopilot.revising --demo --target model --output /tmp/online-model-revision
python -m examples.autopilot.revising --demo --target optimizer --output /tmp/online-optimizer-revision
```

非 demo 还需 `--profile`、`--cases`，以及 `--agent-command` 或 `--llm-model`。profile／cases 的
factory 契约见外层[Elastic Revision](../../../../docs/implementation/elastic-revision.md)。

## 原生训练

使用[任务工作流](../../task_workflows/README.md)或[Hermes 示例](../../hermes/README.md)启动原生
`PPOTrainer`。客户端使用 `resident` 供数；手动调整入口为 `PPOTrainer.reconfigure()`。当前原生自动
Reconfigurator／Revision 接入仍在迁移，显式传入 `elastic_reconfigurator` 会在启动前报错。

完整实现、实验结果与机器操作说明维护在外层[项目文档](../../../../docs/README.md)。
