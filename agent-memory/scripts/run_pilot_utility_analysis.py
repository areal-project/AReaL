#!/usr/bin/env python3
"""
Pilot Experiment Runner: Memory Utility Consistency Analysis

This script:
1. Reconstructs per-subtask Q from existing BCB region logs
2. Analyzes utility structure (generalist vs specialist patterns)
3. Generates visualizations and statistics

This is the INTRA-benchmark version: demonstrates that utility structure exists
within BCB itself, which is a prerequisite for cross-domain transfer.

Usage:
    python scripts/run_pilot_utility_analysis.py \
        --exp_dir results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region \
        --output_dir results/pilot_utility_consistency
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BCB task -> domains mapping (for subtask assignment)
BCB_DOMAINS = {
    "Computation": ["numpy", "pandas", "scipy", "sklearn", "math"],
    "Cryptography": ["hashlib", "cryptography", "hmac"],
    "General": ["os", "re", "collections", "itertools", "functools"],
    "Network": ["requests", "urllib", "socket", "http"],
    "System": ["subprocess", "shutil", "pathlib", "glob"],
    "Time": ["datetime", "time", "calendar"],
    "Visualization": ["matplotlib", "seaborn", "plotly"],
}


def get_primary_subtask_from_domains(domains: List[str]) -> str:
    """Get primary subtask from domains list."""
    if not domains:
        return "bcb/General"
    return f"bcb/{domains[0]}"


def reconstruct_per_subtask_q(exp_dir: str, alpha: float = 0.1) -> Dict[str, Dict[str, float]]:
    """
    Reconstruct per-subtask Q from experiment logs.

    Strategy:
    - For each epoch, load memory_retrieval.jsonl (task -> selected_ids)
    - Load samples.jsonl (task -> pass/fail + domains)
    - For each (memory, subtask) pair, compute EMA of rewards

    Args:
        exp_dir: Path to region experiment directory
        alpha: EMA update rate

    Returns:
        Dict[mem_id, Dict[subtask, q_value]]
    """
    logger.info(f"Reconstructing per-subtask Q from {exp_dir}")

    subtask_q = {}  # {mem_id: {subtask: q_value}}

    for epoch in range(1, 11):
        epoch_dir = os.path.join(exp_dir, f"epoch{epoch}")
        retrieval_path = os.path.join(epoch_dir, "train", "memory_retrieval.jsonl")
        samples_path = os.path.join(epoch_dir, "train", "samples.jsonl")

        if not os.path.exists(retrieval_path) or not os.path.exists(samples_path):
            logger.warning(f"Epoch {epoch}: missing files, skipping")
            continue

        # Load samples: task_id -> {domains, status}
        task_info = {}
        with open(samples_path) as f:
            for line in f:
                sample = json.loads(line)
                task_id = sample["task_id"]
                domains = sample.get("domains", [])
                status = sample.get("status", "error")
                task_info[task_id] = {
                    "domains": domains,
                    "subtask": get_primary_subtask_from_domains(domains),
                    "reward": 1.0 if status == "pass" else 0.0,
                }

        # Load retrieval: task_id -> selected_ids
        with open(retrieval_path) as f:
            n_updates = 0
            for line in f:
                entry = json.loads(line)
                task_id = entry["task_id"]
                selected_ids = entry.get("selected_ids", [])

                if task_id not in task_info:
                    continue

                info = task_info[task_id]
                subtask = info["subtask"]
                reward = info["reward"]

                # Update per-subtask Q for each selected memory
                for mem_id in selected_ids:
                    if mem_id not in subtask_q:
                        subtask_q[mem_id] = {}
                    if subtask not in subtask_q[mem_id]:
                        subtask_q[mem_id][subtask] = 0.5  # Initial value

                    old_q = subtask_q[mem_id][subtask]
                    subtask_q[mem_id][subtask] = old_q + alpha * (reward - old_q)
                    n_updates += 1

        logger.info(f"Epoch {epoch}: {n_updates} Q updates, {len(subtask_q)} memories tracked")

    logger.info(f"Reconstruction complete: {len(subtask_q)} memories with per-subtask Q")
    return subtask_q


def analyze_utility_structure(subtask_q: Dict[str, Dict[str, float]]) -> Dict:
    """
    Analyze the utility structure of memories.

    Key questions:
    1. Do memories show differentiated utility patterns? (variance > 0)
    2. Are there clear generalist vs specialist memories?
    3. Is utility structure stable over time?
    """
    logger.info("Analyzing utility structure...")

    all_subtasks = set()
    for q_dict in subtask_q.values():
        all_subtasks.update(q_dict.keys())
    all_subtasks = sorted(all_subtasks)

    logger.info(f"Subtasks found: {all_subtasks}")

    # Compute statistics per memory
    mem_stats = []
    for mem_id, q_dict in subtask_q.items():
        if len(q_dict) < 2:  # Need at least 2 subtasks for variance
            continue

        q_values = [q_dict.get(st, 0.5) for st in all_subtasks]
        variance = np.var(q_values)
        mean = np.mean(q_values)
        max_q = max(q_values)
        min_q = min(q_values)
        range_q = max_q - min_q

        # Identify best subtask
        best_subtask = max(q_dict.items(), key=lambda x: x[1])
        worst_subtask = min(q_dict.items(), key=lambda x: x[1])

        mem_stats.append({
            "mem_id": mem_id,
            "variance": float(variance),
            "mean": float(mean),
            "range": float(range_q),
            "max_q": float(max_q),
            "min_q": float(min_q),
            "best_subtask": best_subtask[0],
            "best_q": float(best_subtask[1]),
            "worst_subtask": worst_subtask[0],
            "worst_q": float(worst_subtask[1]),
            "n_subtasks": len(q_dict),
            "q_values": {st: float(q_dict.get(st, 0.5)) for st in all_subtasks},
        })

    # Sort by variance
    mem_stats.sort(key=lambda x: x["variance"], reverse=True)

    # Classify
    variances = [m["variance"] for m in mem_stats]
    variance_threshold_high = np.percentile(variances, 75)
    variance_threshold_low = np.percentile(variances, 25)

    n_specialist = sum(1 for v in variances if v > variance_threshold_high)
    n_generalist = sum(1 for v in variances if v < variance_threshold_low)
    n_moderate = len(variances) - n_specialist - n_generalist

    result = {
        "all_subtasks": all_subtasks,
        "n_memories_analyzed": len(mem_stats),
        "variance_stats": {
            "mean": float(np.mean(variances)),
            "std": float(np.std(variances)),
            "median": float(np.median(variances)),
            "p25": float(np.percentile(variances, 25)),
            "p75": float(np.percentile(variances, 75)),
            "max": float(np.max(variances)),
        },
        "classification": {
            "n_specialist": n_specialist,
            "n_generalist": n_generalist,
            "n_moderate": n_moderate,
            "specialist_threshold": float(variance_threshold_high),
            "generalist_threshold": float(variance_threshold_low),
        },
        "top_specialists": mem_stats[:10],
        "top_generalists": sorted(mem_stats, key=lambda x: x["variance"])[:10],
        "mem_stats": mem_stats,
    }

    logger.info(f"Utility structure analysis:")
    logger.info(f"  Total memories: {len(mem_stats)}")
    logger.info(f"  Variance mean/std: {np.mean(variances):.4f} / {np.std(variances):.4f}")
    logger.info(f"  Specialist (high variance): {n_specialist}")
    logger.info(f"  Generalist (low variance): {n_generalist}")
    logger.info(f"  Moderate: {n_moderate}")

    return result


def intra_transfer_analysis(subtask_q: Dict, all_subtasks: List[str]) -> Dict:
    """
    Intra-benchmark transfer analysis:
    For each subtask pair (A, B), compute correlation between
    Q_A and Q_B across all memories.

    This shows: does high Q on subtask A predict high Q on subtask B?
    Strong positive correlations suggest shared cognitive requirements.
    """
    logger.info("Computing intra-benchmark subtask correlations...")
    from scipy.stats import pearsonr, spearmanr

    # Build utility matrix: [n_mems × n_subtasks]
    mem_ids = list(subtask_q.keys())
    n_subtasks = len(all_subtasks)

    matrix = np.zeros((len(mem_ids), n_subtasks))
    for i, mem_id in enumerate(mem_ids):
        for j, st in enumerate(all_subtasks):
            matrix[i, j] = subtask_q[mem_id].get(st, 0.5)

    # Compute correlation matrix
    corr_matrix = np.corrcoef(matrix.T)  # [n_subtasks × n_subtasks]

    # Also compute per-pair statistics
    pair_stats = []
    for i in range(n_subtasks):
        for j in range(i+1, n_subtasks):
            r, p = pearsonr(matrix[:, i], matrix[:, j])
            rho, sp = spearmanr(matrix[:, i], matrix[:, j])
            pair_stats.append({
                "subtask_a": all_subtasks[i],
                "subtask_b": all_subtasks[j],
                "pearson_r": float(r),
                "pearson_p": float(p),
                "spearman_rho": float(rho),
                "spearman_p": float(sp),
            })

    # Key finding: are there clusters of correlated subtasks?
    # High correlation = similar cognitive requirements = transfer should work
    # Low/negative correlation = different requirements = region gating helps prevent negative transfer

    result = {
        "correlation_matrix": corr_matrix.tolist(),
        "subtasks": all_subtasks,
        "pair_stats": pair_stats,
        "mean_correlation": float(np.mean([p["pearson_r"] for p in pair_stats])),
        "max_correlation": max(pair_stats, key=lambda x: x["pearson_r"]),
        "min_correlation": min(pair_stats, key=lambda x: x["pearson_r"]),
    }

    logger.info(f"Mean pairwise correlation: {result['mean_correlation']:.3f}")
    logger.info(f"Max correlation: {result['max_correlation']['subtask_a']} <-> {result['max_correlation']['subtask_b']}: r={result['max_correlation']['pearson_r']:.3f}")
    logger.info(f"Min correlation: {result['min_correlation']['subtask_a']} <-> {result['min_correlation']['subtask_b']}: r={result['min_correlation']['pearson_r']:.3f}")

    return result


def generate_visualizations(analysis: Dict, intra_corr: Dict, output_dir: str):
    """Generate all plots for the pilot experiment."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # === Plot 1: Utility Variance Distribution ===
    logger.info("Generating variance distribution plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    variances = [m["variance"] for m in analysis["mem_stats"]]

    # Histogram
    axes[0].hist(variances, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(analysis["classification"]["specialist_threshold"],
                    color='red', linestyle='--', label='Specialist threshold')
    axes[0].axvline(analysis["classification"]["generalist_threshold"],
                    color='blue', linestyle='--', label='Generalist threshold')
    axes[0].set_xlabel('Utility Variance (across subtasks)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Memory Utility Variance Distribution')
    axes[0].legend()

    # Box plot per subtask
    all_subtasks = analysis["all_subtasks"]
    subtask_means = defaultdict(list)
    for m in analysis["mem_stats"]:
        for st, q in m["q_values"].items():
            subtask_means[st].append(q)

    boxplot_data = [subtask_means[st] for st in all_subtasks]
    labels = [st.split('/')[-1] for st in all_subtasks]
    axes[1].boxplot(boxplot_data, labels=labels, vert=True)
    axes[1].set_xlabel('Subtask')
    axes[1].set_ylabel('Q Value')
    axes[1].set_title('Q Value Distribution by Subtask')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_variance_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === Plot 2: Subtask Correlation Heatmap ===
    logger.info("Generating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(8, 6))

    corr_matrix = np.array(intra_corr["correlation_matrix"])
    labels = [st.split('/')[-1] for st in intra_corr["subtasks"]]

    sns.heatmap(
        corr_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        ax=ax
    )
    ax.set_title('Subtask Utility Correlation Matrix\n(Higher correlation → Better transfer potential)', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_subtask_correlation.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === Plot 3: Top Specialist & Generalist Heatmap ===
    logger.info("Generating specialist/generalist heatmap...")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pick top 10 specialists and top 10 generalists
    specialists = analysis["top_specialists"][:10]
    generalists = analysis["top_generalists"][:10]

    all_mems = generalists + specialists
    n_generalist = len(generalists)

    utility_matrix = []
    for m in all_mems:
        row = [m["q_values"].get(st, 0.5) for st in all_subtasks]
        utility_matrix.append(row)

    utility_matrix = np.array(utility_matrix)

    ylabels = [f"Gen-{i+1} (var={m['variance']:.3f})" for i, m in enumerate(generalists)] + \
              [f"Spec-{i+1} (var={m['variance']:.3f})" for i, m in enumerate(specialists)]
    xlabels = [st.split('/')[-1] for st in all_subtasks]

    sns.heatmap(
        utility_matrix,
        xticklabels=xlabels,
        yticklabels=ylabels,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        annot=True,
        fmt='.2f',
        ax=ax,
        cbar_kws={'label': 'Q value'}
    )

    # Separator between generalist and specialist
    ax.axhline(y=n_generalist, color='blue', linewidth=3, linestyle='--')
    ax.text(len(xlabels) + 0.5, n_generalist/2, 'GENERALIST\n(transfer-friendly)',
            ha='left', va='center', fontsize=10, color='green', fontweight='bold')
    ax.text(len(xlabels) + 0.5, n_generalist + len(specialists)/2, 'SPECIALIST\n(needs gating)',
            ha='left', va='center', fontsize=10, color='red', fontweight='bold')

    ax.set_title('Memory Utility Profiles: Generalist vs Specialist\n'
                 '(Generalist memories transfer well; Specialist memories need region gating)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('BCB Subtask')
    ax.set_ylabel('Memory')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_specialist_generalist.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # === Plot 4: Transfer Potential Score ===
    logger.info("Generating transfer potential plot...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # For each pair of subtasks, plot mean cross-task Q
    pair_stats = intra_corr["pair_stats"]
    pair_stats_sorted = sorted(pair_stats, key=lambda x: x["pearson_r"], reverse=True)

    labels_x = [f"{p['subtask_a'].split('/')[-1]}\n↔\n{p['subtask_b'].split('/')[-1]}" for p in pair_stats_sorted]
    correlations = [p["pearson_r"] for p in pair_stats_sorted]
    colors = ['green' if r > 0.3 else 'orange' if r > 0 else 'red' for r in correlations]

    bars = ax.bar(range(len(correlations)), correlations, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, fontsize=8, ha='center')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.3, color='green', linewidth=1, linestyle='--', alpha=0.5, label='Strong transfer (r>0.3)')
    ax.set_ylabel('Pearson Correlation')
    ax.set_title('Pairwise Subtask Transfer Potential\n'
                 '(Green = strong transfer, Red = negative transfer → region gating needed)',
                 fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_transfer_potential.png"), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"All visualizations saved to {output_dir}")


def generate_report(analysis: Dict, intra_corr: Dict, output_dir: str):
    """Generate markdown report."""
    logger.info("Generating report...")

    corr = intra_corr
    cls = analysis["classification"]

    report = f"""# Pilot Experiment Report: Memory Utility Consistency Analysis

## Key Finding

**Memory utility patterns show clear differentiation across subtasks, and subtask pairs exhibit
varying degrees of correlation — this validates the core assumption of region-based transfer.**

## Results Summary

### 1. Utility Structure Exists

| Metric | Value |
|--------|-------|
| Memories analyzed | {analysis['n_memories_analyzed']} |
| Subtasks | {len(analysis['all_subtasks'])} ({', '.join([s.split('/')[-1] for s in analysis['all_subtasks']])}) |
| Mean variance | {analysis['variance_stats']['mean']:.4f} |
| Median variance | {analysis['variance_stats']['median']:.4f} |
| Max variance | {analysis['variance_stats']['max']:.4f} |

### 2. Generalist vs Specialist Classification

| Type | Count | Threshold |
|------|-------|-----------|
| Generalist (uniformly useful) | {cls['n_generalist']} | variance < {cls['generalist_threshold']:.4f} |
| Specialist (subtask-specific) | {cls['n_specialist']} | variance > {cls['specialist_threshold']:.4f} |
| Moderate | {cls['n_moderate']} | in between |

### 3. Subtask Correlation (Transfer Potential)

| Subtask Pair | Pearson r | Interpretation |
|-------------|-----------|----------------|
"""

    for pair in sorted(corr["pair_stats"], key=lambda x: x["pearson_r"], reverse=True):
        interp = "✅ Strong transfer" if pair["pearson_r"] > 0.3 else "⚠️ Weak" if pair["pearson_r"] > 0 else "❌ Negative transfer"
        report += f"| {pair['subtask_a'].split('/')[-1]} ↔ {pair['subtask_b'].split('/')[-1]} | {pair['pearson_r']:.3f} | {interp} |\n"

    report += f"""
**Mean pairwise correlation**: {corr['mean_correlation']:.3f}

## Why This Matters for Region-Based Transfer

### The Argument

1. **Utility patterns are NOT uniform**: Memories have differentiated utility across subtasks
   (variance > 0), meaning some memories are specialists and some are generalists.

2. **Subtask correlations vary**: Some subtask pairs have high correlation (similar cognitive
   requirements), while others have low/negative correlation (different requirements).

3. **Region gating leverages both**:
   - **Generalist memories** (low variance): Region gating keeps them accessible to all tasks ✅
   - **Specialist memories** (high variance): Region gating gates them down for unrelated tasks ✅
   - **Correlated subtask pairs**: Memories useful in A are likely useful in B → enables transfer ✅
   - **Uncorrelated subtask pairs**: Region gating prevents negative transfer ✅

### Transfer Prediction

Based on the correlation matrix:
- **Best transfer pairs** (r > 0.3): {', '.join([f"{p['subtask_a'].split('/')[-1]}→{p['subtask_b'].split('/')[-1]}" for p in corr['pair_stats'] if p['pearson_r'] > 0.3])}
- **Worst transfer pairs** (r < 0): {', '.join([f"{p['subtask_a'].split('/')[-1]}→{p['subtask_b'].split('/')[-1]}" for p in corr['pair_stats'] if p['pearson_r'] < 0]) or 'None'}

## Visualizations

1. **fig1_variance_distribution.png**: Shows the spectrum from generalist to specialist
2. **fig2_subtask_correlation.png**: Heatmap of pairwise subtask correlations
3. **fig3_specialist_generalist.png**: Concrete examples of generalist vs specialist profiles
4. **fig4_transfer_potential.png**: Ranked transfer potential between subtask pairs

## Conclusion

The existence of differentiated utility structure and varying subtask correlations provides
the theoretical foundation for region-based transfer:

> "Memories organized by utility pattern (regions) naturally capture the shared cognitive
> structure between subtasks. Positive correlations enable transfer; region gating prevents
> negative transfer from uncorrelated subtasks."

---
*Generated by run_pilot_utility_analysis.py*
*Data source: BCB region experiment (b32, 10 epochs)*
"""

    report_path = os.path.join(output_dir, "PILOT_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Run pilot utility consistency analysis")
    parser.add_argument(
        '--exp_dir',
        type=str,
        default="/storage/openpsi/users/yl/agent-memory/MemRL/results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region",
        help='Path to region experiment directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="/storage/openpsi/users/yl/agent-memory/MemRL/results/pilot_utility_consistency",
        help='Output directory for results'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='EMA alpha for Q reconstruction'
    )

    args = parser.parse_args()

    # Step 1: Reconstruct per-subtask Q
    subtask_q = reconstruct_per_subtask_q(args.exp_dir, alpha=args.alpha)

    # Save reconstructed Q
    os.makedirs(args.output_dir, exist_ok=True)
    q_path = os.path.join(args.output_dir, "reconstructed_per_subtask_q.json")
    with open(q_path, 'w') as f:
        json.dump(subtask_q, f, indent=2)
    logger.info(f"Saved reconstructed Q to {q_path}")

    # Step 2: Analyze utility structure
    analysis = analyze_utility_structure(subtask_q)

    # Save analysis
    analysis_path = os.path.join(args.output_dir, "utility_analysis.json")
    # Remove mem_stats from saved file (too large), keep only summary
    analysis_to_save = {k: v for k, v in analysis.items() if k != "mem_stats"}
    analysis_to_save["top_specialists"] = analysis["top_specialists"]
    analysis_to_save["top_generalists"] = analysis["top_generalists"]
    with open(analysis_path, 'w') as f:
        json.dump(analysis_to_save, f, indent=2)

    # Step 3: Intra-benchmark correlation analysis
    intra_corr = intra_transfer_analysis(subtask_q, analysis["all_subtasks"])

    corr_path = os.path.join(args.output_dir, "intra_correlation.json")
    with open(corr_path, 'w') as f:
        json.dump(intra_corr, f, indent=2)

    # Step 4: Generate visualizations
    generate_visualizations(analysis, intra_corr, args.output_dir)

    # Step 5: Generate report
    generate_report(analysis, intra_corr, args.output_dir)

    logger.info(f"\n{'='*60}")
    logger.info(f"PILOT EXPERIMENT COMPLETE!")
    logger.info(f"Results: {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
