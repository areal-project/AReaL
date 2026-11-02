# Elastic 快速命令

在 AReaL 仓库根目录执行。先填好训练配置中的模型、GPU 池、`state_root` 和 `checkpoint_estimated_bytes`。

## 启用 Elastic（原生 Cold）

```bash
CUDA_VISIBLE_DEVICES="" AREAL_ELASTIC_RESOURCE_RECONFIGURATION=1 \
python -m examples.task_workflows.train \
  --config examples/task_workflows/configs/dapo.yaml \
  elastic.enabled=true
```

接入 Reconfigurator 时，追加 `--elastic-profile "$NATIVE_PROFILE"`，指向匹配的原生 Cold 预测配置。

## CPU Simulation

生成 workload：

```bash
python -m areal.elastic.simulation.workload \
  --spec examples/autopilot/elastic/workload-spec.json \
  --output "$WORKLOAD_DIR"
```

指定冻结校准包、组件拟合参数、发布参数和新输出目录，比较 fixed／dynamic：

```bash
python -m examples.autopilot.elastic.simulate_online \
  --reconfigurator reference --mode both --search-workers 4 \
  --workload "$WORKLOAD_DIR/workload.jsonl" \
  --config examples/autopilot/elastic/online-v3-example.json \
  --bundle "$CALIBRATION_BUNDLE" --mechanism-fits "$MECHANISM_FITS" \
  --publication-metadata "$PUBLICATION_METADATA" --output "$OUTPUT"
```

模拟配置中的 `handoff.mode` 选择 Hot／Cold；Cold 需填写 `phase_seconds`。
