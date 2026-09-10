# Visualize MoE expert loads in W&B

Use the `moe_balance/expert_loads` Table for a layer-by-expert heatmap with a step
selector. See [metrics tracking](metrics_tracking.md) for definitions and backend
support.

## Interpret the data

Each Table is one logged-step snapshot with `layer`, `expert`, `tokens`, and
`load_percent`. Layer and expert IDs are zero-based. Tokens count executed routing
assignments, including top-k multiplicity. Percentages sum to 100 per nonempty layer;
empty layers contain zeros.

The configuration below targets Qwen3.5-35B-A3B with **256 routed experts per layer**.
Ideal load is `100 / 256 = 0.390625%`, so the load-to-ideal ratio is
`load_percent * 2.56`. A ratio of 1 means balanced, 2 means twice ideal load, and 0
means no assignments. Fixed color thresholds support comparison across steps:
`[0.8, 1.2)` shares a near-balanced band and values `>= 8` share the darkest band.
Tooltips retain the actual value. For other models, replace `2.56` with `E / 100` and
adjust the expert-axis ticks. Use the number of routed experts, excluding shared
experts; do not use top-k.

## Configure the panel

1. In the project Workspace, select **Add panel → Custom Chart**. Select one run to
   avoid overlapping cells.
1. In the upper-right **Query** area, click **summary ▾**, the gray dropdown below
   `name` and above `keys: [""]` in this editor, and select **historyTable**.
1. Set **tableKey** to `moe_balance/expert_loads`, the logged table key, not a column
   name. Leave `id`, `name`, and `runSets` unchanged.
1. Click **Edit** beside **Line plot** and replace the Vega specification with the JSON
   below.
1. Under **Chart fields**, map each placeholder to its Table column. The previous `x`,
   `y`, and `groupKeys` fields should be replaced by these four fields. Query column
   names may have a `runSets_historyTable_` prefix; choose the actual entries from the
   dropdowns.

| Chart field    | Table column   |
| -------------- | -------------- |
| `layer`        | `layer`        |
| `expert`       | `expert`       |
| `tokens`       | `tokens`       |
| `load_percent` | `load_percent` |

`load_multiple` is calculated inside the chart and needs no mapping. These query and
field-mapping steps follow the
[W&B tutorial](https://docs.wandb.ai/models/app/features/custom-charts/walkthrough).
Enable **Other settings → Show step selector** to switch snapshots. This requires
`historyTable`; `summaryTable` only reads the summary snapshot. See the
[step selector documentation](https://docs.wandb.ai/models/app/features/custom-charts#build-the-graphql-query).
Use **Save as** to save a reusable preset and apply it to the panel.

## Vega specification

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

## Resolve the 10,000-row truncation

Qwen3.5 has 40 layers × 256 experts = **10,240 rows**. The W&B SDK defaults to 10,000
rows for run-media Table serialization, which can remove experts from previews and
charts that consume them. Changing the Vega axis domain cannot recover rows removed
during serialization.

AReaL's `StatsLogger` raises this limit before constructing the expert Table:

```python
wandb.Table.MAX_ROWS = max(wandb.Table.MAX_ROWS, len(expert_rows))
```

Training with this fix preserves all 10,240 rows in subsequent snapshots. The regression
test `test_stats_logger_large_expert_table_serializes_every_layer` checks the actual SDK
serialization, including the final expert. This addresses the SDK limit only: if
downloaded media JSON is complete but the chart is incomplete, inspect the deployed W&B
server's query results and panel limits.

**Existing uploaded previews are not repaired automatically.** Artifact serialization
has a separate limit: the SDK used for validation defaults `MAX_ARTIFACT_ROWS` to
200,000. The old 10,240-row artifacts in this validation were complete even when their
previews were truncated. Raising only `MAX_ARTIFACT_ROWS` does not fix the run-media
`MAX_ROWS` limit.

To recover older snapshots:

1. Download the required Table versions from the source run's Artifacts and inspect the
   `data` row count in `moe_balance/expert_loads.table.json`.
1. If complete, use the artifact for offline visualization or re-log its Table to a
   separate visualization run after raising `MAX_ROWS`. Preserve original log steps when
   recovering multiple snapshots and record source run and artifact versions. Artifact
   version numbers are not necessarily log step numbers.
1. If the artifact is also incomplete, recover from complete local logs or collect the
   data again; increasing limits cannot reconstruct missing rows.

For this Qwen3.5 model, verify 10,240 rows, 40 layers, 256 unique experts per layer, and
the presence of `(layer=39, expert=255)`. Percentages for each nonempty layer should sum
to 100 within floating-point tolerance. For larger tables, also check artifact limits
and server/browser capacity. Postprocessing can split tables by layer while retaining
all experts in each selected layer and the log step. Avoid random sampling followed by
renormalization, which can hide hot experts and change the meaning of balance.

## Pair with scalar history

Plot `moe_balance/layer_<id>/max_over_ideal` beside the heatmap to locate an abnormal
step and layer, then inspect the expert snapshot. Switching snapshots does not require
manual Table merging. To display multiple steps simultaneously in one Table-based chart,
combine snapshots and add a step column in postprocessing.
