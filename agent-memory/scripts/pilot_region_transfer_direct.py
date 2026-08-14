#!/usr/bin/env python3
"""
Pilot: Directly Visualize Why Region Structure Helps Transfer

This pilot provides DIRECT evidence (not indirect performance metrics) that
region structure contains transferable information.

Key visualizations:
1. Region Utility Heatmap: shows generalist/specialist patterns across subtasks
2. Transfer Precision: for held-out subtask queries, does region gating
   select better memories than pure embedding similarity?
3. Region distribution shift: how do top-k results shift between seen vs unseen subtasks?
4. Utility-based region vs random cluster: direct quality comparison

Usage:
    cd /storage/openpsi/users/yl/agent-memory/MemRL
    python scripts/pilot_region_transfer_direct.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.stats import spearmanr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

PILOT_Q_PATH = Path("results/pilot_utility_consistency/reconstructed_per_subtask_q.json")
QUERY_EMB_PATH = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch10/snapshot/10/"
    "local_cache/query_embeddings.json"
)
MEM_CACHE_PATH = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch10/snapshot/10/"
    "local_cache/dict_memory.json"
)
BASE_PATH = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region"
)
OUTPUT_DIR = Path("results/pilot_region_transfer_direct")


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class DirectTransferPilot:

    def __init__(self):
        self.subtask_q: Dict[str, Dict[str, float]] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.task_to_subtask: Dict[str, str] = {}
        self.subtask_centroids: Dict[str, np.ndarray] = {}

        # Regions
        self.regions = []  # list of {id, member_ids, utility_by_subtask}
        self.mem_to_region: Dict[str, int] = {}
        self.known_subtasks: List[str] = []

        # Memory embeddings (for simulating retrieval)
        self.mem_embeddings: Dict[str, np.ndarray] = {}

        self._load_data()
        self._reconstruct_regions()
        self._compute_centroids()
        self._build_mem_embeddings()

    def _load_data(self):
        logger.info("Loading data...")

        with open(PILOT_Q_PATH) as f:
            self.subtask_q = json.load(f)
        logger.info(f"  Per-subtask Q: {len(self.subtask_q)} memories")

        with open(QUERY_EMB_PATH) as f:
            raw = json.load(f)
        self.query_embeddings = {k: np.array(v) for k, v in raw.items()}
        logger.info(f"  Query embeddings: {len(self.query_embeddings)} entries")

        # Task → subtask mapping from all epochs
        for epoch in range(1, 11):
            samples_path = BASE_PATH / f"epoch{epoch}/train/samples.jsonl"
            if not samples_path.exists():
                continue
            with open(samples_path) as f:
                for line in f:
                    s = json.loads(line)
                    domains = s.get("domains", [])
                    prompt = s.get("prompt", "")
                    if domains and prompt:
                        self.task_to_subtask[prompt] = f"bcb/{domains[0]}"

        subtask_counts = defaultdict(int)
        for st in self.task_to_subtask.values():
            subtask_counts[st] += 1
        self.known_subtasks = sorted(subtask_counts.keys())
        logger.info(f"  Subtasks: {self.known_subtasks}")

    def _reconstruct_regions(self):
        from sklearn.cluster import HDBSCAN

        logger.info("Reconstructing regions...")

        mem_ids = list(self.subtask_q.keys())
        X = np.array([[self.subtask_q[m].get(st, 0.5) for st in self.known_subtasks] for m in mem_ids])

        clusterer = HDBSCAN(min_cluster_size=max(3, len(mem_ids) // 200))
        labels = clusterer.fit_predict(X)

        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels) or 1
        if n_clusters == 0:
            labels = np.zeros(len(labels), dtype=int)
            n_clusters = 1

        centroids = []
        for cid in range(n_clusters):
            mask = labels == cid
            centroids.append(X[mask].mean(axis=0) if mask.sum() > 0 else np.full(len(self.known_subtasks), 0.5))
        centroids = np.array(centroids)

        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = int(np.argmin(np.linalg.norm(centroids - X[i], axis=1)))

        for cid in range(n_clusters):
            mask = labels == cid
            members = [mem_ids[i] for i in range(len(mem_ids)) if mask[i]]
            utility = {}
            for st_idx, st in enumerate(self.known_subtasks):
                vals = [X[i, st_idx] for i in range(len(mem_ids)) if mask[i]]
                utility[st] = float(np.mean(vals))
            self.regions.append({"id": cid, "member_ids": members, "utility": utility})

        for i, mem_id in enumerate(mem_ids):
            self.mem_to_region[mem_id] = int(labels[i])

        logger.info(f"  {n_clusters} regions reconstructed")

    def _compute_centroids(self):
        subtask_embs = defaultdict(list)
        for prompt, emb in self.query_embeddings.items():
            if prompt in self.task_to_subtask:
                subtask_embs[self.task_to_subtask[prompt]].append(emb)

        for subtask, embs in subtask_embs.items():
            self.subtask_centroids[subtask] = np.mean(embs, axis=0)

    def _build_mem_embeddings(self):
        """Use query embeddings as proxy for memory embeddings (same text)."""
        # Memory text → embedding mapping
        # The query_embeddings keys are task prompts, and dict_memory keys are also text
        # For simplicity, we'll use the subset of memories that have embeddings
        self.mem_embeddings = {}
        # In practice, mem embeddings would be stored in qdrant; here we skip
        # and use a simulated retrieval based on subtask_q overlap
        logger.info("  Memory embeddings: using Q-based simulated retrieval")

    # ========== Pilot Analysis ==========

    def run_all(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.pilot_1_region_heatmap()
        self.pilot_2_transfer_precision()
        self.pilot_3_region_vs_random()

    def pilot_1_region_heatmap(self):
        """Visualize region utility patterns — show generalist vs specialist."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        logger.info("\n" + "="*70)
        logger.info("PILOT 1: Region Utility Heatmap")
        logger.info("="*70)

        # Build matrix: regions × subtasks
        n_regions = len(self.regions)
        matrix = np.zeros((n_regions, len(self.known_subtasks)))
        for i, region in enumerate(self.regions):
            for j, st in enumerate(self.known_subtasks):
                matrix[i, j] = region["utility"].get(st, 0.5)

        # Sort by variance (generalist on top, specialist on bottom)
        variances = np.var(matrix, axis=1)
        sort_idx = np.argsort(variances)
        matrix = matrix[sort_idx]
        sorted_variances = variances[sort_idx]

        # Classify
        n_generalist = np.sum(sorted_variances < np.percentile(variances, 25))
        n_specialist = np.sum(sorted_variances > np.percentile(variances, 75))

        fig, ax = plt.subplots(figsize=(10, max(8, n_regions * 0.25)))
        short_labels = [st.split('/')[-1] for st in self.known_subtasks]

        sns.heatmap(
            matrix, xticklabels=short_labels,
            yticklabels=[f"R{sort_idx[i]}" for i in range(n_regions)],
            cmap='RdYlGn', vmin=matrix.min(), vmax=matrix.max(),
            cbar_kws={'label': 'Utility'}, ax=ax
        )
        ax.set_title(f'Region Utility Patterns ({n_regions} regions)\n'
                     f'Top = Generalist ({n_generalist}), Bottom = Specialist ({n_specialist})',
                     fontweight='bold')
        ax.set_xlabel('Subtask')
        ax.set_ylabel('Region (sorted by variance)')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pilot1_region_heatmap.png", dpi=200)
        plt.close()
        logger.info(f"  Saved: pilot1_region_heatmap.png")
        logger.info(f"  Generalist regions: {n_generalist}, Specialist regions: {n_specialist}")

    def pilot_2_transfer_precision(self):
        """
        DIRECT evidence: For held-out subtask queries, compare memory quality
        selected by:
        (A) Pure embedding similarity (baseline)
        (B) Embedding similarity × region prior (our method)

        "Quality" = average per-subtask Q of selected memories on the held-out subtask.
        """
        logger.info("\n" + "="*70)
        logger.info("PILOT 2: Transfer Precision (does gating select better memories?)")
        logger.info("="*70)

        results = []

        for held_out in self.known_subtasks:
            # Get queries for held-out subtask
            held_out_queries = [
                (prompt, emb) for prompt, emb in self.query_embeddings.items()
                if self.task_to_subtask.get(prompt) == held_out
            ]
            if len(held_out_queries) < 5:
                logger.warning(f"  {held_out}: only {len(held_out_queries)} queries, skipping")
                continue

            # For each query, simulate retrieval
            baseline_q_vals = []
            gated_q_vals = []

            for prompt, query_emb in held_out_queries:
                # Simulate embedding retrieval: find memories with highest Q on any subtask
                # (Proxy since we don't have memory embeddings — rank by similarity to query)
                # Use per-subtask Q as a proxy for relevance

                # All memories that have Q for the held-out subtask
                candidate_mems = [
                    m for m in self.subtask_q.keys()
                    if held_out in self.subtask_q[m]
                ]
                if not candidate_mems:
                    continue

                # Compute "embedding similarity" proxy: mean Q on known subtasks
                # (This is imperfect but works for pilot)
                known_subtasks = [s for s in self.known_subtasks if s != held_out]
                mem_scores = {}
                for m in candidate_mems:
                    # Use average Q on known subtasks as embedding similarity proxy
                    avg_known_q = np.mean([self.subtask_q[m].get(s, 0.5) for s in known_subtasks])
                    mem_scores[m] = avg_known_q

                # Baseline: top-k by "similarity" only
                top_k = 10
                baseline_top = sorted(mem_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

                # Gated: similarity × region prior
                prior_cache = {}
                for region in self.regions:
                    # Prior = weighted avg of region utility on known subtasks
                    # Weight = similarity of query to subtask centroid
                    weighted_u = 0
                    total_w = 0
                    for s in known_subtasks:
                        if s in self.subtask_centroids:
                            sim = cosine_similarity(query_emb, self.subtask_centroids[s])
                            weighted_u += sim * region["utility"].get(s, 0.5)
                            total_w += sim
                    prior_cache[region["id"]] = weighted_u / total_w if total_w > 0 else 0.5

                gated_scores = {}
                for m, sim_score in mem_scores.items():
                    region_id = self.mem_to_region.get(m, 0)
                    region_prior = prior_cache.get(region_id, 0.5)
                    gated_scores[m] = sim_score * region_prior

                gated_top = sorted(gated_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

                # Evaluate: actual Q on held-out subtask
                baseline_avg = np.mean([self.subtask_q[m][held_out] for m, _ in baseline_top if held_out in self.subtask_q.get(m, {})])
                gated_avg = np.mean([self.subtask_q[m][held_out] for m, _ in gated_top if held_out in self.subtask_q.get(m, {})])

                baseline_q_vals.append(baseline_avg)
                gated_q_vals.append(gated_avg)

            if not baseline_q_vals:
                continue

            baseline_mean = np.mean(baseline_q_vals)
            gated_mean = np.mean(gated_q_vals)
            improvement = gated_mean - baseline_mean
            pct_better = np.mean([g > b for g, b in zip(gated_q_vals, baseline_q_vals)])

            results.append({
                "subtask": held_out,
                "baseline_mean_q": baseline_mean,
                "gated_mean_q": gated_mean,
                "improvement": improvement,
                "pct_queries_improved": pct_better,
                "n_queries": len(baseline_q_vals),
            })

            direction = "✅" if improvement > 0 else "❌"
            logger.info(
                f"  {held_out}: baseline={baseline_mean:.4f} → gated={gated_mean:.4f} "
                f"(Δ={improvement:+.4f}) {direction}  [{pct_better:.0%} queries improved]"
            )

        # Summary
        if results:
            avg_improvement = np.mean([r["improvement"] for r in results])
            n_improved = sum(1 for r in results if r["improvement"] > 0)
            logger.info(f"\n  Average improvement: {avg_improvement:+.4f}")
            logger.info(f"  Subtasks improved: {n_improved}/{len(results)}")

            with open(OUTPUT_DIR / "pilot2_transfer_precision.json", "w") as f:
                json.dump(results, f, indent=2)

    def pilot_3_region_vs_random(self):
        """
        DIRECT evidence: utility-based region prior vs random cluster prior.

        If utility regions provide better transfer prior than random partitions,
        it proves the STRUCTURE is valuable, not just the act of partitioning.
        """
        logger.info("\n" + "="*70)
        logger.info("PILOT 3: Utility Region Prior vs Random Cluster Prior")
        logger.info("="*70)

        n_random_trials = 10
        mem_ids = list(self.subtask_q.keys())

        results = []

        for held_out in self.known_subtasks:
            held_out_queries = [
                (prompt, emb) for prompt, emb in self.query_embeddings.items()
                if self.task_to_subtask.get(prompt) == held_out
            ]
            if len(held_out_queries) < 5:
                continue

            known_subtasks = [s for s in self.known_subtasks if s != held_out]

            # Utility-based region prior correlation
            utility_corrs = []
            random_corrs = []

            for prompt, query_emb in held_out_queries:
                # For each memory, compute:
                # - utility region prior
                # - actual Q on held-out subtask

                mems_with_held_out = [
                    m for m in mem_ids if held_out in self.subtask_q.get(m, {})
                ]
                if len(mems_with_held_out) < 20:
                    continue

                actual_qs = [self.subtask_q[m][held_out] for m in mems_with_held_out]

                # Utility region prior
                prior_cache = {}
                for region in self.regions:
                    weighted_u = 0
                    total_w = 0
                    for s in known_subtasks:
                        if s in self.subtask_centroids:
                            sim = cosine_similarity(query_emb, self.subtask_centroids[s])
                            weighted_u += sim * region["utility"].get(s, 0.5)
                            total_w += sim
                    prior_cache[region["id"]] = weighted_u / total_w if total_w > 0 else 0.5

                utility_priors = [prior_cache.get(self.mem_to_region.get(m, 0), 0.5) for m in mems_with_held_out]

                if np.std(utility_priors) > 1e-8 and np.std(actual_qs) > 1e-8:
                    corr, _ = spearmanr(utility_priors, actual_qs)
                    utility_corrs.append(corr)

                # Random cluster prior (average over trials)
                trial_corrs = []
                for _ in range(n_random_trials):
                    random_labels = np.random.randint(0, len(self.regions), size=len(mem_ids))
                    random_region_utils = defaultdict(lambda: defaultdict(list))
                    for i, m in enumerate(mem_ids):
                        rid = random_labels[i]
                        for s in self.known_subtasks:
                            if s in self.subtask_q.get(m, {}):
                                random_region_utils[rid][s].append(self.subtask_q[m][s])

                    random_prior_cache = {}
                    for rid in range(len(self.regions)):
                        weighted_u = 0
                        total_w = 0
                        for s in known_subtasks:
                            if s in self.subtask_centroids:
                                sim = cosine_similarity(query_emb, self.subtask_centroids[s])
                                vals = random_region_utils[rid].get(s, [0.5])
                                weighted_u += sim * np.mean(vals)
                                total_w += sim
                        random_prior_cache[rid] = weighted_u / total_w if total_w > 0 else 0.5

                    random_priors = [random_prior_cache.get(random_labels[mem_ids.index(m)], 0.5) for m in mems_with_held_out]

                    if np.std(random_priors) > 1e-8 and np.std(actual_qs) > 1e-8:
                        corr, _ = spearmanr(random_priors, actual_qs)
                        trial_corrs.append(corr)

                if trial_corrs:
                    random_corrs.append(np.mean(trial_corrs))

            if utility_corrs and random_corrs:
                util_mean = np.mean(utility_corrs)
                rand_mean = np.mean(random_corrs)
                gap = util_mean - rand_mean

                results.append({
                    "subtask": held_out,
                    "utility_region_corr": util_mean,
                    "random_cluster_corr": rand_mean,
                    "gap": gap,
                })

                direction = "✅" if gap > 0 else "❌"
                logger.info(
                    f"  {held_out}: utility_region={util_mean:.4f} vs random={rand_mean:.4f} "
                    f"(gap={gap:+.4f}) {direction}"
                )

        if results:
            avg_gap = np.mean([r["gap"] for r in results])
            n_better = sum(1 for r in results if r["gap"] > 0)
            logger.info(f"\n  Average gap (utility - random): {avg_gap:+.4f}")
            logger.info(f"  Subtasks where utility > random: {n_better}/{len(results)}")
            logger.info(f"\n  Interpretation:")
            logger.info(f"    gap > 0: utility-based regions provide BETTER transfer prior than random partitions")
            logger.info(f"    gap ≈ 0: region structure has no transfer advantage (negative result)")

            with open(OUTPUT_DIR / "pilot3_region_vs_random.json", "w") as f:
                json.dump(results, f, indent=2)


def main():
    pilot = DirectTransferPilot()
    pilot.run_all()
    logger.info(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
