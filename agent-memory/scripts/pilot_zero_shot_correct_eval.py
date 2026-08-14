#!/usr/bin/env python3
"""
Pilot: Zero-Shot Utility Predictor with Correct Evaluation

Validates:
1. Assumption: region utility on subtask ≈ region utility on individual queries
2. Predictor performance with per-query evaluation

Usage:
    cd /storage/openpsi/users/yl/agent-memory/MemRL
    python scripts/pilot_zero_shot_correct_eval.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
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
BASE_PATH = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region"
)
OUTPUT_DIR = Path("results/pilot_zero_shot_correct_eval")


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Region:
    def __init__(self, region_id: int):
        self.region_id = region_id
        self.utility_by_subtask: Dict[str, float] = {}
        self.member_ids: List[str] = []


class PilotCorrectEval:

    def __init__(self):
        self.subtask_q: Dict[str, Dict[str, float]] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.task_data: Dict[str, dict] = {}  # {(task_id, epoch): {subtask, reward, prompt, selected_mems}}
        self.subtask_centroids: Dict[str, np.ndarray] = {}
        self.regions: List[Region] = []
        self.mem_to_region: Dict[str, int] = {}
        self.known_subtasks: List[str] = []

        self._load_data()
        self._reconstruct_regions()
        self._compute_centroids()

    def _load_data(self):
        logger.info("Loading data...")

        # 1. Per-subtask Q
        with open(PILOT_Q_PATH) as f:
            self.subtask_q = json.load(f)
        logger.info(f"  Per-subtask Q: {len(self.subtask_q)} memories")

        # 2. Query embeddings
        with open(QUERY_EMB_PATH) as f:
            raw = json.load(f)
        self.query_embeddings = {k: np.array(v) for k, v in raw.items()}
        logger.info(f"  Query embeddings: {len(self.query_embeddings)} entries")

        # 3. Load task data from all epochs
        for epoch in range(1, 11):
            samples_path = BASE_PATH / f"epoch{epoch}/train/samples.jsonl"
            retrieval_path = BASE_PATH / f"epoch{epoch}/train/memory_retrieval.jsonl"

            if not samples_path.exists() or not retrieval_path.exists():
                continue

            # Load retrieval
            retrieval = {}
            with open(retrieval_path) as f:
                for line in f:
                    entry = json.loads(line)
                    retrieval[entry["task_id"]] = entry.get("selected_ids", [])

            # Load samples
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
        for st in self.known_subtasks:
            logger.info(f"    {st}: {subtask_counts[st]}")

    def _reconstruct_regions(self):
        """Re-cluster regions from subtask_q."""
        from sklearn.cluster import HDBSCAN

        logger.info("Reconstructing regions...")

        mem_ids = []
        utility_matrix = []

        for mem_id, q_dict in self.subtask_q.items():
            mem_ids.append(mem_id)
            vec = [q_dict.get(st, 0.5) for st in self.known_subtasks]
            utility_matrix.append(vec)

        X = np.array(utility_matrix)
        min_cluster_size = max(3, len(mem_ids) // 200)
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        labels = clusterer.fit_predict(X)

        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            labels = np.zeros(len(labels), dtype=int)
            n_clusters = 1

        # Compute centroids and assign noise
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

        logger.info(f"  Clustered into {n_clusters} regions")

        # Build regions
        for cid in range(n_clusters):
            region = Region(cid)
            mask = labels == cid
            region.member_ids = [mem_ids[i] for i in range(len(mem_ids)) if mask[i]]

            for st_idx, st in enumerate(self.known_subtasks):
                utils = [X[i, st_idx] for i in range(len(mem_ids)) if mask[i]]
                region.utility_by_subtask[st] = float(np.mean(utils)) if utils else 0.5

            self.regions.append(region)

        for i, mem_id in enumerate(mem_ids):
            self.mem_to_region[mem_id] = int(labels[i])

    def _compute_centroids(self):
        """Compute subtask centroids."""
        subtask_embs = defaultdict(list)

        for info in self.task_data.values():
            prompt = info["prompt"]
            if prompt in self.query_embeddings:
                subtask_embs[info["subtask"]].append(self.query_embeddings[prompt])

        for subtask, embs in subtask_embs.items():
            self.subtask_centroids[subtask] = np.mean(embs, axis=0)

    def validate_assumption(self):
        """Validate: region utility on subtask ≈ region utility on queries."""
        logger.info("\n" + "="*70)
        logger.info("VALIDATING ASSUMPTION: subtask-level utility ≈ query-level utility")
        logger.info("="*70)

        results = []

        for region in self.regions:
            query_utilities = defaultdict(list)

            # Collect per-query utilities
            for info in self.task_data.values():
                subtask = info["subtask"]
                selected_mems = info["selected_mems"]

                # Memory from this region
                region_mems = [m for m in selected_mems if self.mem_to_region.get(m) == region.region_id]

                if region_mems:
                    # Average Q of these memories
                    q_vals = [self.subtask_q[m][subtask] for m in region_mems if m in self.subtask_q]
                    if q_vals:
                        query_utilities[subtask].append(np.mean(q_vals))

            # Compute variances
            intra_vars = []
            for subtask, utilities in query_utilities.items():
                if len(utilities) > 1:
                    intra_vars.append(np.var(utilities))

            if not intra_vars:
                continue

            mean_intra_var = np.mean(intra_vars)

            subtask_means = [np.mean(utilities) for utilities in query_utilities.values() if utilities]
            if len(subtask_means) < 2:
                continue

            inter_var = np.var(subtask_means)
            ratio = inter_var / mean_intra_var if mean_intra_var > 1e-8 else 0

            results.append({
                "region_id": region.region_id,
                "intra_var": mean_intra_var,
                "inter_var": inter_var,
                "ratio": ratio,
            })

        # Summary
        if results:
            mean_ratio = np.mean([r["ratio"] for r in results])
            logger.info(f"\nMean inter/intra variance ratio: {mean_ratio:.2f}")
            logger.info(f"  > 5: Strong assumption (subtask-level is good proxy)")
            logger.info(f"  2-5: Moderate assumption")
            logger.info(f"  < 2: Weak assumption (need query-level modeling)")

            with open(OUTPUT_DIR / "assumption_validation.json", "w") as f:
                json.dump(results, f, indent=2)

        return results

    def run_correct_eval(self):
        """Evaluate predictor with per-query ground truth."""
        logger.info("\n" + "="*70)
        logger.info("CORRECT EVALUATION: per-query prediction")
        logger.info("="*70)

        results = {"elementwise": []}

        for held_out in self.known_subtasks:
            if held_out not in self.subtask_centroids:
                continue

            logger.info(f"\nHeld-out: {held_out}")

            train_subtasks = [s for s in self.known_subtasks if s != held_out]

            # Training data
            train_X, train_y = [], []

            for info in self.task_data.values():
                if info["subtask"] == held_out:
                    continue

                prompt = info["prompt"]
                if prompt not in self.query_embeddings:
                    continue

                query_emb = self.query_embeddings[prompt]
                psi = self._compute_psi(query_emb, train_subtasks)

                for mem_id in info["selected_mems"]:
                    if mem_id not in self.mem_to_region or mem_id not in self.subtask_q:
                        continue

                    region_id = self.mem_to_region[mem_id]
                    region = self.regions[region_id]

                    region_pattern = np.array([region.utility_by_subtask.get(st, 0.5) for st in train_subtasks])
                    features = np.concatenate([region_pattern, psi, region_pattern * psi])

                    label = self.subtask_q[mem_id][info["subtask"]]

                    train_X.append(features)
                    train_y.append(label)

            # Test data
            test_X, test_y = [], []

            for info in self.task_data.values():
                if info["subtask"] != held_out:
                    continue

                prompt = info["prompt"]
                if prompt not in self.query_embeddings:
                    continue

                query_emb = self.query_embeddings[prompt]
                psi = self._compute_psi(query_emb, train_subtasks)

                for mem_id in info["selected_mems"]:
                    if mem_id not in self.mem_to_region or mem_id not in self.subtask_q:
                        continue

                    region_id = self.mem_to_region[mem_id]
                    region = self.regions[region_id]

                    region_pattern = np.array([region.utility_by_subtask.get(st, 0.5) for st in train_subtasks])
                    features = np.concatenate([region_pattern, psi, region_pattern * psi])

                    label = self.subtask_q[mem_id][held_out]

                    test_X.append(features)
                    test_y.append(label)

            if len(test_y) < 5 or len(train_y) < 10:
                logger.warning(f"  Not enough data: train={len(train_y)}, test={len(test_y)}")
                continue

            train_X, train_y = np.array(train_X), np.array(train_y)
            test_X, test_y = np.array(test_X), np.array(test_y)

            logger.info(f"  Train: {len(train_y)}, Test: {len(test_y)}")

            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            model.fit(train_X, train_y)

            pred = model.predict(test_X)
            pred = np.clip(pred, 0, 1)

            metrics = self._evaluate(pred, test_y)
            results["elementwise"].append(metrics)

            logger.info(f"  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}  Spearman={metrics['spearman']:.4f}")

        self._print_summary(results)
        return results

    def _compute_psi(self, query_emb: np.ndarray, subtasks: List[str]) -> np.ndarray:
        sims = []
        for st in subtasks:
            if st in self.subtask_centroids:
                sims.append(cosine_similarity(query_emb, self.subtask_centroids[st]))
            else:
                sims.append(0.0)
        return np.array(sims)

    def _evaluate(self, pred: np.ndarray, true: np.ndarray) -> dict:
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        spearman = 0.0
        if len(pred) >= 3 and np.std(pred) > 1e-8 and np.std(true) > 1e-8:
            spearman, _ = spearmanr(pred, true)

        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "spearman": float(spearman)}

    def _print_summary(self, results: dict):
        logger.info("\n" + "="*70)
        logger.info("SUMMARY")
        logger.info("="*70)

        errors = results["elementwise"]
        if errors:
            mae = np.mean([e["mae"] for e in errors])
            r2 = np.mean([e["r2"] for e in errors])
            spearman = np.mean([e["spearman"] for e in errors])

            logger.info(f"Elementwise: MAE={mae:.4f}  R²={r2:.4f}  Spearman={spearman:.4f}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot = PilotCorrectEval()

    # 1. Validate assumption
    pilot.validate_assumption()

    # 2. Correct evaluation
    results = pilot.run_correct_eval()

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
