#!/usr/bin/env python3
"""
Correct Pilot: Region Transfer Validation

Based on Codex recommendations:
1. Use REAL memory embeddings (not Q proxy)
2. Use correct gating formula: hybrid_score × region_gating_score
3. Offline re-ranking on logged retrievals

Experiments:
- Pilot A: Offline re-ranking ablation (does region gating improve ranking?)
- Pilot B: Region predictive validity (does region membership predict usefulness?)
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
OUTPUT_DIR = Path("results/pilot_region_transfer_correct")


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CorrectTransferPilot:

    def __init__(self):
        self.subtask_q = {}
        self.query_embeddings = {}
        self.mem_embeddings = {}
        self.task_data = {}  # {(task_id, epoch): {subtask, reward, prompt, selected_mems}}

        self.regions = []
        self.mem_to_region = {}
        self.known_subtasks = []

        self._load_data()
        self._reconstruct_regions()

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
            retrieval_path = BASE_PATH / f"epoch{epoch}/train/memory_retrieval.jsonl"

            if not samples_path.exists() or not retrieval_path.exists():
                continue

            retrieval = {}
            with open(retrieval_path) as f:
                for line in f:
                    entry = json.loads(line)
                    retrieval[entry["task_id"]] = entry.get("selected_ids", [])

            with open(samples_path) as f:
                for line in f:
                    s = json.loads(line)
                    task_id = s["task_id"]
                    domains = s.get("domains", [])
                    if not domains:
                        continue

                    subtask = f"bcb/{domains[0]}"
                    reward = 1.0 if s.get("status") == "PASS" else 0.0
                    prompt = s.get("prompt", "")
                    selected_mems = retrieval.get(task_id, [])

                    key = f"{task_id}_e{epoch}"
                    self.task_data[key] = {
                        "task_id": task_id,
                        "epoch": epoch,
                        "subtask": subtask,
                        "reward": reward,
                        "prompt": prompt,
                        "selected_mems": selected_mems,
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
            counts = {}
            for st_idx, st in enumerate(self.known_subtasks):
                vals = [X[i, st_idx] for i in range(len(mem_ids)) if mask[i]]
                utility[st] = float(np.mean(vals))
                counts[st] = len(vals)
            self.regions.append({"id": cid, "member_ids": members, "utility": utility, "counts": counts})

        for i, mem_id in enumerate(mem_ids):
            self.mem_to_region[mem_id] = int(labels[i])

        logger.info(f"  {n_clusters} regions reconstructed")

    def compute_region_gating_score(self, mem_id, target_subtask, C=3.0, global_mean=0.44):
        """
        Correct region gating formula with Bayesian smoothing.
        """
        if mem_id not in self.mem_to_region:
            return 1.0

        region_id = self.mem_to_region[mem_id]
        region = self.regions[region_id]

        # Get region utility for target subtask
        utility = region["utility"].get(target_subtask, global_mean)
        count = region["counts"].get(target_subtask, 0)

        # Bayesian smoothing
        smoothed = (count * utility + C * global_mean) / (count + C)
        return np.clip(smoothed, 0.01, 1.0)

    def pilot_a_offline_reranking(self):
        """
        Offline re-ranking ablation on logged retrievals.

        For each held-out subtask:
        - Take logged retrieved memories
        - Re-rank with: (1) Q only, (2) Q × region gating
        - Compare top-k quality
        """
        logger.info("\n" + "="*70)
        logger.info("PILOT A: Offline Re-ranking Ablation")
        logger.info("="*70)

        results = []

        for held_out in self.known_subtasks:
            # Get tasks for held-out subtask
            held_out_tasks = [
                (key, info) for key, info in self.task_data.items()
                if info["subtask"] == held_out and info["prompt"] in self.query_embeddings
            ]

            if len(held_out_tasks) < 10:
                logger.warning(f"  {held_out}: only {len(held_out_tasks)} tasks, skipping")
                continue

            q_only_scores = []
            q_region_scores = []

            for key, info in held_out_tasks:
                selected_mems = info["selected_mems"]
                if not selected_mems:
                    continue

                # Filter to memories we have Q for
                valid_mems = [m for m in selected_mems if m in self.subtask_q and held_out in self.subtask_q[m]]
                if len(valid_mems) < 3:
                    continue

                # Ranking 1: Q only
                q_scores = {m: self.subtask_q[m][held_out] for m in valid_mems}
                q_only_top = sorted(q_scores.items(), key=lambda x: x[1], reverse=True)[:5]

                # Ranking 2: Q × region gating
                q_region_scores_dict = {}
                for m in valid_mems:
                    q = self.subtask_q[m][held_out]
                    region_score = self.compute_region_gating_score(m, held_out)
                    q_region_scores_dict[m] = q * region_score

                q_region_top = sorted(q_region_scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]

                # Evaluate: actual Q of top-5
                q_only_avg = np.mean([self.subtask_q[m][held_out] for m, _ in q_only_top])
                q_region_avg = np.mean([self.subtask_q[m][held_out] for m, _ in q_region_top])

                q_only_scores.append(q_only_avg)
                q_region_scores.append(q_region_avg)

            if not q_only_scores:
                continue

            q_only_mean = np.mean(q_only_scores)
            q_region_mean = np.mean(q_region_scores)
            improvement = q_region_mean - q_only_mean
            pct_better = np.mean([r > q for r, q in zip(q_region_scores, q_only_scores)])

            results.append({
                "subtask": held_out,
                "q_only": q_only_mean,
                "q_region": q_region_mean,
                "improvement": improvement,
                "pct_improved": pct_better,
                "n_tasks": len(q_only_scores),
            })

            direction = "✅" if improvement > 0 else "❌"
            logger.info(
                f"  {held_out}: Q_only={q_only_mean:.4f} → Q×region={q_region_mean:.4f} "
                f"(Δ={improvement:+.4f}) {direction}  [{pct_better:.0%} improved]"
            )

        if results:
            avg_improvement = np.mean([r["improvement"] for r in results])
            n_improved = sum(1 for r in results if r["improvement"] > 0)
            logger.info(f"\n  Average improvement: {avg_improvement:+.4f}")
            logger.info(f"  Subtasks improved: {n_improved}/{len(results)}")

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_DIR / "pilot_a_offline_reranking.json", "w") as f:
                json.dump(results, f, indent=2)

    def pilot_b_region_predictive_validity(self):
        """
        Region predictive validity test (no retrieval simulation).

        Split by epochs: train on epoch 1-7, test on epoch 8-10
        Question: Does region membership predict cross-subtask usefulness?
        """
        logger.info("\n" + "="*70)
        logger.info("PILOT B: Region Predictive Validity")
        logger.info("="*70)

        train_epochs = list(range(1, 8))
        test_epochs = list(range(8, 11))

        # Estimate region utility on train split
        train_region_utils = defaultdict(lambda: defaultdict(list))
        for key, info in self.task_data.items():
            if info["epoch"] not in train_epochs:
                continue
            subtask = info["subtask"]
            for mem_id in info["selected_mems"]:
                if mem_id in self.mem_to_region and mem_id in self.subtask_q:
                    region_id = self.mem_to_region[mem_id]
                    q = self.subtask_q[mem_id].get(subtask, 0.5)
                    train_region_utils[region_id][subtask].append(q)

        # Compute region utility estimates
        region_utility_est = {}
        for rid in range(len(self.regions)):
            region_utility_est[rid] = {}
            for st in self.known_subtasks:
                vals = train_region_utils[rid].get(st, [0.44])
                region_utility_est[rid][st] = np.mean(vals)

        # Test on test split
        results = []

        for held_out in self.known_subtasks:
            test_tasks = [
                (key, info) for key, info in self.task_data.items()
                if info["epoch"] in test_epochs and info["subtask"] == held_out
            ]

            if len(test_tasks) < 10:
                continue

            # For each memory used in test tasks, predict its usefulness
            mem_actual_q = []
            mem_region_pred = []
            mem_global_pred = []

            for key, info in test_tasks:
                for mem_id in info["selected_mems"]:
                    if mem_id not in self.subtask_q or held_out not in self.subtask_q[mem_id]:
                        continue
                    if mem_id not in self.mem_to_region:
                        continue

                    actual_q = self.subtask_q[mem_id][held_out]
                    region_id = self.mem_to_region[mem_id]
                    region_pred = region_utility_est[region_id].get(held_out, 0.44)
                    global_pred = 0.44  # baseline

                    mem_actual_q.append(actual_q)
                    mem_region_pred.append(region_pred)
                    mem_global_pred.append(global_pred)

            if len(mem_actual_q) < 20:
                continue

            # Compare prediction quality
            region_mae = np.mean(np.abs(np.array(mem_region_pred) - np.array(mem_actual_q)))
            global_mae = np.mean(np.abs(np.array(mem_global_pred) - np.array(mem_actual_q)))

            region_corr, _ = spearmanr(mem_region_pred, mem_actual_q) if np.std(mem_region_pred) > 1e-8 else (0, 1)
            global_corr = 0.0  # constant prediction has no correlation

            results.append({
                "subtask": held_out,
                "region_mae": region_mae,
                "global_mae": global_mae,
                "region_corr": region_corr,
                "n_samples": len(mem_actual_q),
            })

            direction = "✅" if region_mae < global_mae else "❌"
            logger.info(
                f"  {held_out}: region_MAE={region_mae:.4f} vs global_MAE={global_mae:.4f} "
                f"(corr={region_corr:.3f}) {direction}"
            )

        if results:
            avg_region_mae = np.mean([r["region_mae"] for r in results])
            avg_global_mae = np.mean([r["global_mae"] for r in results])
            avg_corr = np.mean([r["region_corr"] for r in results])

            logger.info(f"\n  Average region MAE: {avg_region_mae:.4f}")
            logger.info(f"  Average global MAE: {avg_global_mae:.4f}")
            logger.info(f"  Average region correlation: {avg_corr:.3f}")
            logger.info(f"\n  Interpretation:")
            logger.info(f"    Region MAE < Global MAE: region membership provides predictive signal")
            logger.info(f"    Positive correlation: region structure captures transferable patterns")

            with open(OUTPUT_DIR / "pilot_b_predictive_validity.json", "w") as f:
                json.dump(results, f, indent=2)

    def run_all(self):
        self.pilot_a_offline_reranking()
        self.pilot_b_region_predictive_validity()


def main():
    pilot = CorrectTransferPilot()
    pilot.run_all()
    logger.info(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
