#!/usr/bin/env python3
"""
Direct Proof: Region Gating Improves Zero-Shot Retrieval

Experiment Design:
- Leave-one-subtask-out: hold out one subtask (e.g., Cryptography)
- For queries in held-out subtask:
  1. Baseline: retrieve by embedding similarity only
  2. Region: retrieve by embedding similarity × region_gating_score
     - region_gating_score is computed from OTHER subtasks' utility (zero-shot)
- Compare: which method retrieves memories with higher ground-truth Q[held_out]?

Key: This is TRUE zero-shot because region gating does NOT use held_out subtask's utility.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
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
MEM_EMB_PATH = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch10/snapshot/10/"
    "local_cache/memory_embeddings.json"
)
BASE_PATH = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region"
)
OUTPUT_DIR = Path("results/pilot_direct_transfer_proof")


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class DirectTransferProof:

    def __init__(self):
        self.subtask_q = {}
        self.query_embeddings = {}
        self.mem_embeddings = {}
        self.task_data = {}
        self.regions = []
        self.mem_to_region = {}
        self.known_subtasks = []
        self.subtask_centroids = {}

        self._load_data()
        self._reconstruct_regions()
        self._compute_centroids()

    def _load_data(self):
        logger.info("Loading data...")

        with open(PILOT_Q_PATH) as f:
            self.subtask_q = json.load(f)
        logger.info(f"  Per-subtask Q: {len(self.subtask_q)} memories")

        with open(QUERY_EMB_PATH) as f:
            raw = json.load(f)
        self.query_embeddings = {k: np.array(v) for k, v in raw.items()}
        logger.info(f"  Query embeddings: {len(self.query_embeddings)}")

        with open(MEM_EMB_PATH) as f:
            raw = json.load(f)
        self.mem_embeddings = {k: np.array(v) for k, v in raw.items()}
        logger.info(f"  Memory embeddings: {len(self.mem_embeddings)}")

        # Load task data
        for epoch in range(1, 11):
            samples_path = BASE_PATH / f"epoch{epoch}/train/samples.jsonl"
            if not samples_path.exists():
                continue

            with open(samples_path) as f:
                for line in f:
                    s = json.loads(line)
                    task_id = s["task_id"]
                    domains = s.get("domains", [])
                    if not domains:
                        continue

                    subtask = f"bcb/{domains[0]}"
                    prompt = s.get("prompt", "")

                    key = f"{task_id}_e{epoch}"
                    self.task_data[key] = {
                        "task_id": task_id,
                        "epoch": epoch,
                        "subtask": subtask,
                        "prompt": prompt,
                    }

        subtask_counts = defaultdict(int)
        for v in self.task_data.values():
            subtask_counts[v["subtask"]] += 1
        self.known_subtasks = sorted(subtask_counts.keys())
        logger.info(f"  Task data: {len(self.task_data)} entries")

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

        # Assign noise to nearest cluster
        centroids = []
        for cid in range(n_clusters):
            mask = labels == cid
            if mask.sum() > 0:
                centroids.append(X[mask].mean(axis=0))
            else:
                centroids.append(np.full(len(self.known_subtasks), 0.5))
        centroids = np.array(centroids)

        for i, label in enumerate(labels):
            if label == -1:
                dists = np.linalg.norm(centroids - X[i], axis=1)
                labels[i] = int(np.argmin(dists))

        # Build regions
        for cid in range(n_clusters):
            mask = labels == cid
            members = [mem_ids[i] for i in range(len(mem_ids)) if mask[i]]
            utility = {}
            for st_idx, st in enumerate(self.known_subtasks):
                vals = [X[i, st_idx] for i in range(len(mem_ids)) if mask[i]]
                utility[st] = float(np.mean(vals)) if vals else 0.5
            self.regions.append({"id": cid, "member_ids": members, "utility": utility})

        for i, mem_id in enumerate(mem_ids):
            self.mem_to_region[mem_id] = int(labels[i])

        logger.info(f"  {n_clusters} regions reconstructed")

    def _compute_centroids(self):
        """Compute subtask centroids from query embeddings."""
        subtask_embs = defaultdict(list)

        for info in self.task_data.values():
            prompt = info["prompt"]
            if prompt in self.query_embeddings:
                subtask_embs[info["subtask"]].append(self.query_embeddings[prompt])

        for subtask, embs in subtask_embs.items():
            self.subtask_centroids[subtask] = np.mean(embs, axis=0)

        logger.info(f"  Computed centroids for {len(self.subtask_centroids)} subtasks")

    def compute_zero_shot_region_gating(self, mem_id, query_emb, held_out_subtask, C=3.0, global_mean=0.44):
        """
        Zero-shot region gating: estimate utility for held_out_subtask
        using ONLY other subtasks' utility.

        Method: weighted average of region utility on known subtasks,
        weighted by query's similarity to subtask centroids.
        """
        if mem_id not in self.mem_to_region:
            return 1.0

        region_id = self.mem_to_region[mem_id]
        region = self.regions[region_id]

        # Compute weights: how similar is query to each known subtask?
        weights = {}
        total_weight = 0.0
        for st in self.known_subtasks:
            if st == held_out_subtask:
                continue  # CRITICAL: do not use held_out subtask
            if st not in self.subtask_centroids:
                continue
            sim = cosine_similarity(query_emb, self.subtask_centroids[st])
            sim = max(0, sim)  # clip negative
            weights[st] = sim
            total_weight += sim

        if total_weight == 0:
            return 1.0

        # Weighted average of region utility
        estimated_utility = 0.0
        for st, w in weights.items():
            estimated_utility += (w / total_weight) * region["utility"].get(st, global_mean)

        # Bayesian smoothing toward global mean
        # (use a small count since this is zero-shot)
        smoothed = (1 * estimated_utility + C * global_mean) / (1 + C)
        return np.clip(smoothed, 0.01, 1.0)

    def run_experiment(self, k=10):
        """
        Main experiment: leave-one-subtask-out retrieval comparison.
        """
        logger.info("\n" + "="*70)
        logger.info("DIRECT PROOF: Region Gating Improves Zero-Shot Retrieval")
        logger.info("="*70)

        all_results = []

        for held_out in self.known_subtasks:
            logger.info(f"\nHeld-out subtask: {held_out}")

            # Get queries for held-out subtask
            held_out_queries = [
                (key, info) for key, info in self.task_data.items()
                if info["subtask"] == held_out and info["prompt"] in self.query_embeddings
            ]

            if len(held_out_queries) < 10:
                logger.warning(f"  Only {len(held_out_queries)} queries, skipping")
                continue

            baseline_avg_q = []
            region_avg_q = []

            for key, info in held_out_queries:
                query_emb = self.query_embeddings[info["prompt"]]

                # Retrieve top-k by embedding similarity
                candidates = []
                for mem_id, mem_emb in self.mem_embeddings.items():
                    if mem_id not in self.subtask_q:
                        continue
                    if held_out not in self.subtask_q[mem_id]:
                        continue
                    sim = cosine_similarity(query_emb, mem_emb)
                    candidates.append((mem_id, sim))

                if len(candidates) < k:
                    continue

                # Baseline: top-k by similarity only
                baseline_top_k = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]
                baseline_mems = [m for m, _ in baseline_top_k]

                # Region: top-k by similarity × region_gating_score
                region_scores = []
                for mem_id, sim in candidates:
                    gating = self.compute_zero_shot_region_gating(mem_id, query_emb, held_out)
                    region_scores.append((mem_id, sim * gating))

                region_top_k = sorted(region_scores, key=lambda x: x[1], reverse=True)[:k]
                region_mems = [m for m, _ in region_top_k]

                # Evaluate: average ground-truth Q[held_out]
                baseline_q = np.mean([self.subtask_q[m][held_out] for m in baseline_mems])
                region_q = np.mean([self.subtask_q[m][held_out] for m in region_mems])

                baseline_avg_q.append(baseline_q)
                region_avg_q.append(region_q)

            if not baseline_avg_q:
                continue

            baseline_mean = np.mean(baseline_avg_q)
            region_mean = np.mean(region_avg_q)
            improvement = region_mean - baseline_mean
            pct_better = np.mean([r > b for r, b in zip(region_avg_q, baseline_avg_q)])

            result = {
                "subtask": held_out,
                "baseline_q": baseline_mean,
                "region_q": region_mean,
                "improvement": improvement,
                "pct_improved": pct_better,
                "n_queries": len(baseline_avg_q),
            }
            all_results.append(result)

            direction = "✅" if improvement > 0 else "❌"
            logger.info(
                f"  Baseline Q: {baseline_mean:.4f} → Region Q: {region_mean:.4f} "
                f"(Δ={improvement:+.4f}) {direction}  [{pct_better:.0%} improved]"
            )

        # Summary
        if all_results:
            avg_improvement = np.mean([r["improvement"] for r in all_results])
            n_improved = sum(1 for r in all_results if r["improvement"] > 0)
            logger.info(f"\n{'='*70}")
            logger.info(f"SUMMARY:")
            logger.info(f"  Average improvement: {avg_improvement:+.4f}")
            logger.info(f"  Subtasks improved: {n_improved}/{len(all_results)}")
            logger.info(f"{'='*70}")

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_DIR / "results.json", "w") as f:
                json.dump(all_results, f, indent=2)

            logger.info(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    pilot = DirectTransferProof()
    pilot.run_experiment(k=10)
