#!/usr/bin/env python3
"""Parse wandb binary (.wandb) from a verl GRPO training run and generate visualizations.

Reads the protobuf history records directly to get all metrics including
metrics/parallel_ratio, val/parallel_ratio, val-core/*, etc.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import wandb.proto.wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore

WANDB_FILE = Path(__file__).parent / "run-20260412_233219-ui4hnf40/run-ui4hnf40.wandb"
OUT_DIR = Path(__file__).parent

# ─── 1. Parse wandb binary ────────────────────────────────────────────────────
print("Parsing wandb binary...")
ds = DataStore()
ds.open_for_scan(str(WANDB_FILE))

history_rows = []
while True:
    data = ds.scan_data()
    if data is None:
        break
    try:
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            row = {}
            for item in rec.history.item:
                full_key = "/".join(item.nested_key) if item.nested_key else item.key
                if not full_key:
                    continue
                try:
                    val = json.loads(item.value_json)
                    if isinstance(val, (int, float)):
                        row[full_key] = val
                except:
                    pass
            history_rows.append(row)
    except:
        pass

df = pd.DataFrame(history_rows)
df = df.sort_values("training/global_step").reset_index(drop=True)
steps = df["training/global_step"]
print(f"Parsed {len(df)} steps: {steps.min():.0f} → {steps.max():.0f}")

# Validation subset (every 10 steps)
val_mask = df["val-core/math_dapo/reward/mean@8"].notna()
val_df = df[val_mask].copy()
val_steps = val_df["training/global_step"]
print(f"Validation points: {len(val_df)} at steps {val_steps.tolist()}")

# ─── 2. Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "grid.linewidth": 0.5,
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 8,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

C = {
    "blue":    "#58a6ff",
    "orange":  "#f78166",
    "green":   "#3fb950",
    "purple":  "#d2a8ff",
    "amber":   "#f0883e",
    "cyan":    "#79c0ff",
    "lime":    "#56d364",
    "red":     "#ff7b72",
    "pink":    "#f778ba",
    "teal":    "#39d2c0",
}


def smooth(y, window=5):
    """Simple moving average for noisy curves."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    y_arr = np.array(y, dtype=float)
    # Handle NaN
    mask = np.isnan(y_arr)
    y_arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y_arr[~mask]) if (~mask).any() else 0
    padded = np.pad(y_arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_with_smooth(ax, x, y, color, label, window=5, alpha_raw=0.25, lw_smooth=2.0):
    """Plot raw + smoothed curve, handling NaN gaps properly."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() == 0:
        return
    ax.plot(x[valid], y[valid], color=color, alpha=alpha_raw, linewidth=0.8)
    y_smooth = smooth(y[valid], window)
    ax.plot(x[valid], y_smooth, color=color, linewidth=lw_smooth, label=label)


# ─── 3. Figure 1: Core Metrics (requested) ────────────────────────────────────
fig1, axes = plt.subplots(4, 3, figsize=(20, 18))
fig1.suptitle(
    f"GRPO Training — Fix-ckpt-saving-6  (6×8 NVIDIA L20X, verl/GRPO, {int(steps.max())} steps)",
    fontsize=15, fontweight="bold", y=0.995,
)

# --- Row 1: Parallel ratios (train) ---
# 1a: metrics/parallel_ratio
ax = axes[0, 0]
plot_with_smooth(ax, steps, df["metrics/parallel_ratio"], C["blue"], "parallel_ratio")
ax.set_title("metrics/parallel_ratio")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True)

# 1b: metrics/trial_parallel_ratio
ax = axes[0, 1]
plot_with_smooth(ax, steps, df["metrics/trial_parallel_ratio"], C["green"], "trial_parallel_ratio")
ax.set_title("metrics/trial_parallel_ratio")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True)

# 1c: metrics/subtask_parallel_ratio
ax = axes[0, 2]
plot_with_smooth(ax, steps, df["metrics/subtask_parallel_ratio"], C["purple"], "subtask_parallel_ratio")
ax.set_title("metrics/subtask_parallel_ratio")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True)

# --- Row 2: Validation parallel ratios + val reward ---
# 2a: val/parallel_ratio
ax = axes[1, 0]
ax.plot(val_steps, val_df["val/parallel_ratio"], color=C["blue"], linewidth=2, marker="o", markersize=6, label="val/parallel_ratio")
ax.set_title("val/parallel_ratio")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True)
ax.legend()

# 2b: val/trial_parallel_ratio
ax = axes[1, 1]
ax.plot(val_steps, val_df["val/trial_parallel_ratio"], color=C["green"], linewidth=2, marker="o", markersize=6, label="val/trial_parallel_ratio")
ax.set_title("val/trial_parallel_ratio")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True)
ax.legend()

# 2c: val/subtask_parallel_ratio
ax = axes[1, 2]
ax.plot(val_steps, val_df["val/subtask_parallel_ratio"], color=C["purple"], linewidth=2, marker="o", markersize=6, label="val/subtask_parallel_ratio")
ax.set_title("val/subtask_parallel_ratio")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True)
ax.legend()

# --- Row 3: val-core reward, critic/score, response_length ---
# 3a: val-core/math_dapo/reward/mean@8
ax = axes[2, 0]
ax.plot(val_steps, val_df["val-core/math_dapo/reward/mean@8"], color=C["orange"], linewidth=2, marker="s", markersize=6, label="mean@8")
if "val-core/math_dapo/reward/best@8/mean" in val_df.columns:
    ax.plot(val_steps, val_df["val-core/math_dapo/reward/best@8/mean"], color=C["green"], linewidth=2, marker="^", markersize=6, label="best@8")
ax.set_title("val-core/math_dapo/reward")
ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.grid(True)
ax.legend()

# 3b: critic/score/mean
ax = axes[2, 1]
plot_with_smooth(ax, steps, df["critic/score/mean"], C["cyan"], "critic/score/mean")
ax.set_title("critic/score/mean")
ax.set_xlabel("Step")
ax.set_ylabel("Score")
ax.grid(True)
ax.legend()

# 3c: response_length/mean
ax = axes[2, 2]
plot_with_smooth(ax, steps, df["response_length/mean"], C["amber"], "response_length/mean")
ax.set_title("response_length/mean")
ax.set_xlabel("Step")
ax.set_ylabel("Tokens")
ax.grid(True)
ax.legend()

# --- Row 4: reward_extra/num_tokens_in_the_longest_thread/mean ---
ax = axes[3, 0]
plot_with_smooth(ax, steps, df["reward_extra/num_tokens_in_the_longest_thread/mean"], C["teal"], "longest_thread tokens")
ax.set_title("reward_extra/num_tokens_in_the_longest_thread/mean")
ax.set_xlabel("Step")
ax.set_ylabel("Tokens")
ax.grid(True)
ax.legend()

# Hide unused axes in row 4
axes[3, 1].set_visible(False)
axes[3, 2].set_visible(False)

fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig(OUT_DIR / "01_core_metrics.png", dpi=150, bbox_inches="tight")
print("Saved 01_core_metrics.png")


# ─── 4. Figure 2: Overlay comparisons ─────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle("Training vs Validation Comparison", fontsize=15, fontweight="bold", y=0.995)

# 2a: All parallel ratios (train) overlaid
ax = axes2[0, 0]
plot_with_smooth(ax, steps, df["metrics/parallel_ratio"], C["blue"], "parallel_ratio")
plot_with_smooth(ax, steps, df["metrics/trial_parallel_ratio"], C["green"], "trial_parallel_ratio")
plot_with_smooth(ax, steps, df["metrics/subtask_parallel_ratio"], C["purple"], "subtask_parallel_ratio")
ax.set_title("Train: All Parallel Ratios")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True)
ax.legend()

# 2b: All parallel ratios (val) overlaid
ax = axes2[0, 1]
ax.plot(val_steps, val_df["val/parallel_ratio"], color=C["blue"], linewidth=2, marker="o", markersize=6, label="parallel_ratio")
ax.plot(val_steps, val_df["val/trial_parallel_ratio"], color=C["green"], linewidth=2, marker="s", markersize=6, label="trial_parallel_ratio")
ax.plot(val_steps, val_df["val/subtask_parallel_ratio"], color=C["purple"], linewidth=2, marker="^", markersize=6, label="subtask_parallel_ratio")
ax.set_title("Val: All Parallel Ratios")
ax.set_xlabel("Step")
ax.set_ylabel("Ratio")
ax.grid(True)
ax.legend()

# 2c: Reward: train critic/score/mean vs val-core reward
ax = axes2[1, 0]
plot_with_smooth(ax, steps, df["critic/score/mean"], C["cyan"], "train: critic/score/mean")
ax.plot(val_steps, val_df["val-core/math_dapo/reward/mean@8"], color=C["orange"], linewidth=2.5, marker="o", markersize=7, label="val: mean@8", zorder=5)
ax.set_title("Train Score vs Val Reward")
ax.set_xlabel("Step")
ax.set_ylabel("Reward / Score")
ax.grid(True)
ax.legend()

# 2d: Actor loss + entropy (dual axis)
ax = axes2[1, 1]
plot_with_smooth(ax, steps, df["actor/pg_loss"], C["red"], "pg_loss")
ax.axhline(0, color="#8b949e", linewidth=0.5, linestyle="--")
ax.set_xlabel("Step")
ax.set_ylabel("Policy Loss", color=C["red"])
ax.tick_params(axis="y", labelcolor=C["red"])
ax.grid(True)

ax_r = ax.twinx()
plot_with_smooth(ax_r, steps, df["actor/entropy"], C["teal"], "entropy")
ax_r.set_ylabel("Entropy", color=C["teal"])
ax_r.tick_params(axis="y", labelcolor=C["teal"])
ax.set_title("Actor: Policy Loss & Entropy")

# Manual legend for twin axes
lines = [
    Line2D([0], [0], color=C["red"], lw=2, label="pg_loss"),
    Line2D([0], [0], color=C["teal"], lw=2, label="entropy"),
]
ax.legend(handles=lines, loc="upper right")

fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig(OUT_DIR / "02_comparison.png", dpi=150, bbox_inches="tight")
print("Saved 02_comparison.png")


# ─── 5. Figure 3: Performance & timing ────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 11))
fig3.suptitle("Performance & Timing", fontsize=15, fontweight="bold", y=0.995)

# 3a: Step timing breakdown
ax = axes3[0, 0]
for col, label, color in [
    ("timing_s/step", "Total Step", C["blue"]),
    ("timing_s/gen", "Generation", C["orange"]),
    ("timing_s/update_actor", "Actor Update", C["green"]),
    ("timing_s/old_log_prob", "Old Log Prob", C["purple"]),
]:
    if col in df.columns:
        plot_with_smooth(ax, steps, df[col], color, label, window=3)
ax.set_title("Timing Breakdown (seconds)")
ax.set_xlabel("Step")
ax.set_ylabel("Seconds")
ax.grid(True)
ax.legend()

# 3b: MFU
ax = axes3[0, 1]
if "perf/mfu/actor" in df.columns:
    plot_with_smooth(ax, steps, df["perf/mfu/actor"], C["cyan"], "MFU (actor)")
ax.set_title("Model FLOPs Utilization (Actor)")
ax.set_xlabel("Step")
ax.set_ylabel("MFU (%)")
ax.grid(True)
ax.legend()

# 3c: GPU Memory
ax = axes3[1, 0]
if "perf/max_memory_allocated_gb" in df.columns:
    plot_with_smooth(ax, steps, df["perf/max_memory_allocated_gb"], C["red"], "Max Allocated")
if "perf/max_memory_reserved_gb" in df.columns:
    plot_with_smooth(ax, steps, df["perf/max_memory_reserved_gb"], C["amber"], "Max Reserved")
ax.set_title("GPU Memory (GB)")
ax.set_xlabel("Step")
ax.set_ylabel("GB")
ax.grid(True)
ax.legend()

# 3d: Grad norm + advantages
ax = axes3[1, 1]
plot_with_smooth(ax, steps, df["actor/grad_norm"], C["lime"], "grad_norm")
ax.set_xlabel("Step")
ax.set_ylabel("Grad Norm", color=C["lime"])
ax.tick_params(axis="y", labelcolor=C["lime"])
ax.grid(True)

ax_r = ax.twinx()
plot_with_smooth(ax_r, steps, df["critic/advantages/mean"], C["pink"], "advantages/mean")
ax_r.axhline(0, color="#8b949e", linewidth=0.5, linestyle="--")
ax_r.set_ylabel("Advantages Mean", color=C["pink"])
ax_r.tick_params(axis="y", labelcolor=C["pink"])
ax.set_title("Grad Norm & Advantages")
lines = [
    Line2D([0], [0], color=C["lime"], lw=2, label="grad_norm"),
    Line2D([0], [0], color=C["pink"], lw=2, label="advantages/mean"),
]
ax.legend(handles=lines, loc="upper right")

fig3.tight_layout(rect=[0, 0, 1, 0.97])
fig3.savefig(OUT_DIR / "03_performance.png", dpi=150, bbox_inches="tight")
print("Saved 03_performance.png")


# ─── 5b. Figure 4: val-core/math_dapo/reward/mean@8 (standalone) ─────────────
fig4, ax4 = plt.subplots(figsize=(10, 6))
fig4.suptitle("val-core/math_dapo/reward/mean@8", fontsize=15, fontweight="bold", y=0.995)

ax4.plot(val_steps, val_df["val-core/math_dapo/reward/mean@8"], color=C["orange"],
         linewidth=2, marker="o", markersize=7, label="mean@8")
ax4.set_ylim(0.71, 0.8)
ax4.set_xlabel("Step")
ax4.set_ylabel("Reward")
ax4.grid(True)
ax4.legend()

fig4.tight_layout(rect=[0, 0, 1, 0.97])
fig4.savefig(OUT_DIR / "04_val_reward_mean8.png", dpi=150, bbox_inches="tight")
print("Saved 04_val_reward_mean8.png")


# ─── 6. Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("TRAINING RUN SUMMARY")
print("=" * 72)
print(f"  Experiment:  Fix-ckpt-saving-6 (deepscaler)")
print(f"  Algorithm:   GRPO  |  Framework: verl + HuggingFace 4.51.1")
print(f"  Hardware:    6×8 NVIDIA L20X (Hopper, ~140GB)  |  CUDA 12.8")
print(f"  Steps:       {steps.min():.0f} → {steps.max():.0f} of 12,480")
print()

def fmt_range(col):
    s = df[col]
    return f"{s.iloc[0]:.4f} → {s.iloc[-1]:.4f} (min={s.min():.4f}, max={s.max():.4f})"

print(f"  metrics/parallel_ratio:          {fmt_range('metrics/parallel_ratio')}")
print(f"  metrics/trial_parallel_ratio:    {fmt_range('metrics/trial_parallel_ratio')}")
print(f"  metrics/subtask_parallel_ratio:  {fmt_range('metrics/subtask_parallel_ratio')}")
print()
print(f"  val/parallel_ratio:              {val_df['val/parallel_ratio'].iloc[0]:.4f} → {val_df['val/parallel_ratio'].iloc[-1]:.4f}")
print(f"  val/trial_parallel_ratio:        {val_df['val/trial_parallel_ratio'].iloc[0]:.4f} → {val_df['val/trial_parallel_ratio'].iloc[-1]:.4f}")
print(f"  val/subtask_parallel_ratio:      {val_df['val/subtask_parallel_ratio'].iloc[0]:.4f} → {val_df['val/subtask_parallel_ratio'].iloc[-1]:.4f}")
print()
print(f"  val-core/math_dapo/reward/mean@8: {val_df['val-core/math_dapo/reward/mean@8'].iloc[0]:.4f} → {val_df['val-core/math_dapo/reward/mean@8'].iloc[-1]:.4f} (best={val_df['val-core/math_dapo/reward/mean@8'].max():.4f})")
print(f"  val-core/math_dapo/reward/best@8: {val_df['val-core/math_dapo/reward/best@8/mean'].iloc[0]:.4f} → {val_df['val-core/math_dapo/reward/best@8/mean'].iloc[-1]:.4f} (best={val_df['val-core/math_dapo/reward/best@8/mean'].max():.4f})")
print()
print(f"  critic/score/mean:               {fmt_range('critic/score/mean')}")
print(f"  response_length/mean:            {fmt_range('response_length/mean')}")
print(f"  actor/entropy:                   {fmt_range('actor/entropy')}")
print()
print(f"  Avg step time: {df['timing_s/step'].mean():.0f}s (~{df['timing_s/step'].mean()/60:.1f} min)")
print("=" * 72)
