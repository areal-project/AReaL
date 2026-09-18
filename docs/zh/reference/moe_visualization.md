# 在 W&B 中可视化 MoE 专家负载

本文使用 `moe_balance/expert_loads` Table 制作“层 × 专家”热力图，并用 step 滑块切换训练快照。
指标定义和后端支持见[指标跟踪](metrics_tracking.md)。

## 数据与颜色含义

每个 Table 是一个日志 step 的快照，包含 `layer`、`expert`、`tokens`、`load_percent` 四列。 层和专家编号从 0
开始；`tokens` 表示实际路由分配数，包含 top-k 重复分配。 非空层的 `load_percent` 合计为 100；空层为 0。

下面的配置用于每层 **256 个专家**的 Qwen3.5-35B-A3B：

- 理想占比为 `100 / 256 = 0.390625%`。
- 相对理想负载为 `load_percent / (100 / 256) = load_percent * 2.56`。
- `1×` 表示均衡，`2×` 表示理想负载的两倍，`0×` 表示该专家没有路由分配。
- 固定阈值颜色可用于跨 step 比较；`[0.8, 1.2)` 共用一个接近均衡的色阶，`>= 8×` 共用最深色。 最深色不代表数值被截断，悬停仍可查看实际倍数。

其他模型应将 `2.56` 改为实际专家数 `E / 100`，并调整横轴刻度。 这里的专家数仅包含 routed experts，不包含 shared
experts，也不是每个 token 的 top-k。

## 配置面板

1. 在 W&B 项目 Workspace 中选择 **Add panel → Custom Chart**。先只选择一个 run，避免不同 run 的单元格重叠。
1. 点击右上角 **Query** 区域的 **summary ▾**：在此界面中，它是 `name` 下面、`keys: [""]` 上面的灰色下拉框。 将其切换为
   **historyTable**。这是
   [W&B 官方教程](https://docs.wandb.ai/models/app/features/custom-charts/walkthrough)中的查询入口。
1. 在出现的 **tableKey** 中填写 `moe_balance/expert_loads`。 填的是表名，不是 `layer` 或 `expert`
   等列名；`id`、`name` 和上面的 `runSets` 保持原样。
1. 点击左上角 **Line plot** 旁边的 **Edit**，将 Vega 配置完整替换为下一节的 JSON。
1. 回到右侧 **Chart fields**。原有的 `x`、`y`、`groupKeys` 应变成配置定义的四个字段，按下表从下拉框选择对应列。 列名可能带有
   `runSets_historyTable_` 前缀，以实际下拉框为准。

| 图表字段       | 选择的 Table 列 |
| -------------- | --------------- |
| `layer`        | `layer`         |
| `expert`       | `expert`        |
| `tokens`       | `tokens`        |
| `load_percent` | `load_percent`  |

`load_multiple` 是图表内部计算的字段，不需要映射。 在 **Other settings** 中开启 **Show step
selector**，通过滑块选择日志 step。 该选项需要 `historyTable`；`summaryTable` 只读取 summary 中的快照。
参见[官方 step 滑块说明](https://docs.wandb.ai/models/app/features/custom-charts#build-the-graphql-query)。
最后使用 **Save as** 保存 preset，再应用到面板。

## 可直接粘贴的 Vega 配置

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
  "data": {
    "name": "wandb"
  },
  "title": "MoE expert load — 1× = balanced",
  "transform": [
    {
      "calculate": "datum['${field:load_percent}'] * 2.56",
      "as": "load_multiple"
    }
  ],
  "mark": "rect",
  "encoding": {
    "x": {
      "field": "${field:expert}",
      "type": "ordinal",
      "sort": "ascending",
      "title": "Expert",
      "axis": {
        "values": [
          0,
          32,
          64,
          96,
          128,
          160,
          192,
          224,
          255
        ],
        "labelAngle": 0
      },
      "scale": {
        "paddingInner": 0,
        "paddingOuter": 0
      }
    },
    "y": {
      "field": "${field:layer}",
      "type": "ordinal",
      "sort": "ascending",
      "title": "Layer",
      "scale": {
        "paddingInner": 0,
        "paddingOuter": 0
      }
    },
    "color": {
      "field": "load_multiple",
      "type": "quantitative",
      "title": "Load / ideal (×)",
      "scale": {
        "type": "threshold",
        "domain": [
          0.25,
          0.5,
          0.8,
          1.2,
          2,
          4,
          8
        ],
        "range": [
          "#fff7ec",
          "#fee8c8",
          "#fdd49e",
          "#fdbb84",
          "#fc8d59",
          "#ef6548",
          "#b30000",
          "#7f0000"
        ]
      },
      "legend": {
        "orient": "right",
        "format": ".2~f"
      }
    },
    "tooltip": [
      {
        "field": "${field:layer}",
        "type": "ordinal",
        "title": "Layer"
      },
      {
        "field": "${field:expert}",
        "type": "ordinal",
        "title": "Expert"
      },
      {
        "field": "${field:tokens}",
        "type": "quantitative",
        "title": "Tokens",
        "format": ",.2~f"
      },
      {
        "field": "load_multiple",
        "type": "quantitative",
        "title": "Load / ideal (×)",
        "format": ".3f"
      }
    ]
  },
  "config": {
    "view": {
      "stroke": "#d1d5db"
    },
    "axis": {
      "grid": false
    }
  }
}
```

## 排查 10,000 行截断

Qwen3.5 的 40 层 × 256 个专家共有 **10,240 行**。W&B SDK 的 run-media Table 序列化默认上限为 10,000
行，可能导致最后一层的部分专家从预览及依赖该预览的图表中消失。 这是数据序列化阶段的截断，修改 Vega 横轴范围无法找回这些行。

AReaL 的 `StatsLogger` 已在创建专家 Table 前按实际行数提高上限：

```python
wandb.Table.MAX_ROWS = max(wandb.Table.MAX_ROWS, len(expert_rows))
```

使用包含此修复的训练版本，后续快照会保留全部 10,240 行。回归测试
`test_stats_logger_large_expert_table_serializes_every_layer` 检查实际 SDK 序列化结果的行数和最后一个专家。
这只处理 SDK 序列化上限；如果导出的 media JSON 完整而图表仍缺行，需进一步检查当前 W&B 服务端的查询结果和面板限制。

**已上传的旧快照不会自动修复。** Table artifact 与 run-media 预览使用不同的序列化上限； 当前验证环境的 SDK
中，`MAX_ARTIFACT_ROWS` 默认为 200,000，因此这次 10,240 行的旧 artifact 完整，预览却可能截断。 仅提高
`MAX_ARTIFACT_ROWS` 不能解决 run-media 的 `MAX_ROWS` 限制。

处理旧数据时：

1. 从原 run 的 Artifacts 中下载需要的 Table 版本，检查其中 `moe_balance/expert_loads.table.json` 的
   `data` 行数。
1. 若 artifact 完整，可直接用它做离线热力图，或在设置 `MAX_ROWS` 后将完整 Table 重新上报到独立的可视化 run。 恢复多个快照时保留原日志
   step，并记录源 run 和 artifact 版本；artifact 的版本号不一定等于日志 step。
1. 若 artifact 本身也缺行，提高上限不能恢复丢失数据，需从完整本地日志恢复或重新采集。

对这份 Qwen3.5 数据，完整性检查应包括：10,240 行、40 个层编号、每层 256 个唯一专家， 以及 `(layer=39, expert=255)`
存在。非空层占比之和应在浮点误差范围内等于 100%。

专家规模更大时，还需检查 SDK 的 artifact 上限以及服务端和浏览器的处理能力。 可以在后处理中按层拆分 Table，每张表保留选定层的所有专家，并保留日志
step。 不要随机抽样后重新归一化，否则会掩盖热点专家并改变均衡度含义。

## 配合标量定位问题

在热力图旁展示 `moe_balance/layer_<id>/max_over_ideal` 折线：先定位异常 step 和层，
再用专家热力图查看该步的具体负载。切换快照无需手动合并 Table； 若要把多个 step 同时绘制在同一张图中，则需要在后处理中合并快照并添加 step 列。
