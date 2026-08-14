#!/usr/bin/env python3
"""
Pilot: Zero-Shot Utility Predictor with Reconstructed Regions

Uses old checkpoint data:
  - reconstructed per-subtask Q (from pilot_utility_consistency)
  - query embeddings (from snapshot)
  - Re-cluster regions from subtask Q on-the-fly

Validates:
  - Elementwise interaction features
  - Bilinear features (ablation)
  - Leave-one-subtask-out cross-validation

Usage:
    cd /storage/openpsi/users/yl/agent-memory/MemRL
    python scripts/pilot_zero_shot_with_reconstructed_regions.py
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.stats import spearmanr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PILOT_Q_PATH = Path("results/pilot_utility_consistency/reconstructed_per_subtask_q.json")
QUERY_EMB_PATH = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch10/snapshot/10/"
    "local_cache/query_embeddings.json"
)
SAMPLES_PATH = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch10/train/samples.jsonl"
)
OUTPUT_DIR = Path("results/pilot_zero_shot_reconstructed")


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Region:
    """Simple region structure."""
    def __init__(self, region_id: int):
        self.region_id = region_id
        self.utility_by_subtask: Dict[str, float] = {}
        self.member_ids: List[str] = []


class PilotZeroShotReconstructed:

    def __init__(self):
        self.subtask_q: Dict[str, Dict[str, float]] = {}
        self.query_embeddings: Dict[str, np.ndarray] = {}
        self.task_to_subtask: Dict[str, str] = {}
        self.subtask_centroids: Dict[str, np.ndarray] = {}
        self.regions: List[Region] = []
        self.mem_to_region: Dict[str, int] = {}  # Hard assignment for simplicity
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

        # 3. Task → subtask mapping
        with open(SAMPLES_PATH) as f:
            for line in f:
                s = json.loads(line)
                domains = s.get("domains", [])
                if domains:
                    prompt = s.get("prompt", "")
                    self.task_to_subtask[prompt] = f"bcb/{domains[0]}"

        subtask_counts = defaultdict(int)
        for st in self.task_to_subtask.values():
            subtask_counts[st] += 1
        self.known_subtasks = sorted(subtask_counts.keys())
        logger.info(f"  Known subtasks: {self.known_subtasks}")

    def _reconstruct_regions(self):
        """Re-cluster regions from subtask_q using HDBSCAN."""
        from sklearn.cluster import HDBSCAN

        logger.info("Reconstructing regions from per-subtask Q...")

        # Build utility matrix
        mem_ids = []
        utility_matrix = []

        for mem_id, q_dict in self.subtask_q.items():
            mem_ids.append(mem_id)
            vec = [q_dict.get(st, 0.5) for st in self.known_subtasks]
            utility_matrix.append(vec)

        X = np.array(utility_matrix)
        logger.info(f"  Utility matrix: {X.shape}")

        # HDBSCAN clustering
        min_cluster_size = max(3, len(mem_ids) // 200)
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
        )
        labels = clusterer.fit_predict(X)

        # Assign noise to nearest cluster
        unique_labels = set(labels) - {-1}
        n_clusters = len(unique_labels)

        if n_clusters == 0:
            logger.warning("No clusters found, creating single region")
            labels = np.zeros(len(labels), dtype=int)
            n_clusters = 1

        # Compute centroids
        centroids = []
        for cid in range(n_clusters):
            mask = labels == cid
            if mask.sum() > 0:
                centroids.append(X[mask].mean(axis=0))
            else:
                centroids.append(np.full(len(self.known_subtasks), 0.5))
        centroids = np.array(centroids)

        # Assign noise to nearest
        for i, label in enumerate(labels):
            if label == -1:
                dists = np.linalg.norm(centroids - X[i], axis=1)
                labels[i] = int(np.argmin(dists))

        logger.info(f"  Clustered into {n_clusters} regions")

        # Build region structures
        for cid in range(n_clusters):
            region = Region(cid)
            mask = labels == cid
            region.member_ids = [mem_ids[i] for i in range(len(mem_ids)) if mask[i]]

            # Compute region utility per subtask
            for st_idx, st in enumerate(self.known_subtasks):
                utils = [X[i, st_idx] for i in range(len(mem_ids)) if mask[i]]
                region.utility_by_subtask[st] = float(np.mean(utils)) if utils else 0.5

            self.regions.append(region)
            logger.info(f"    Region {cid}: {len(region.member_ids)} members")

        # Build mem → region mapping (hard assignment)
        for i, mem_id in enumerate(mem_ids):
            self.mem_to_region[mem_id] = int(labels[i])

    def _compute_centroids(self):
        """Compute subtask centroids from query embeddings."""
        subtask_embs = defaultdict(list)

        for prompt, emb in self.query_embeddings.items():
            if prompt in self.task_to_subtask:
                subtask = self.task_to_subtask[prompt]
                subtask_embs[subtask].append(emb)

        for subtask, embs in subtask_embs.items():
            self.subtask_centroids[subtask] = np.mean(embs, axis=0)
            logger.info(f"  Centroid for {subtask}: {len(embs)} queries")

    def _compute_psi(self, query_emb: np.ndarray, exclude_subtask: str = None) -> np.ndarray:
        """ψ = [sim(query_emb, centroid_st) for st in known_subtasks, excluding held-out]"""
        sims = []
        for st in self.known_subtasks:
            if st == exclude_subtask:
                continue
            if st not in self.subtask_centroids:
                sims.append(0.0)
            else:
                sims.append(cosine_similarity(query_emb, self.subtask_centroids[st]))
        return np.array(sims)

    def _build_features(
        self, region_pattern: np.ndarray, psi: np.ndarray, method: str
    ) -> np.ndarray:
        """Build feature vector."""
        if method == "baseline":
            # Just use mean region utility (no query info)
            return np.array([region_pattern.mean()])
        elif method == "elementwise":
            return np.concatenate([region_pattern, psi, region_pattern * psi])
        elif method == "bilinear":
            outer = np.outer(region_pattern, psi).flatten()
            return np.concatenate([region_pattern, psi, outer])
        else:
            raise ValueError(f"Unknown method: {method}")

    def run_leave_one_out(self):
        """Leave-one-subtask-out validation."""
        results = {
            "baseline": [],
            "elementwise": [],
            "bilinear": [],
        }

        for held_out in self.known_subtasks:
            if held_out not in self.subtask_centroids:
                logger.warning(f"No centroid for {held_out}, skipping")
                continue

            logger.info(f"\n{'='*70}")
            logger.info(f"Held-out subtask: {held_out}")
            logger.info(f"{'='*70}")

            train_subtasks = [s for s in self.known_subtasks if s != held_out]

            # Collect training samples: (region, subtask) → utility
            train_X = {"baseline": [], "elementwise": [], "bilinear": []}
            train_y = []

            for region in self.regions:
                # Skip if region has no data on held_out subtask
                if held_out not in region.utility_by_subtask:
                    continue

                # Region pattern (mask held_out)
                region_pattern = np.array([
                    region.utility_by_subtask.get(st, 0.5)
                    for st in train_subtasks
                ])

                # For each task in held_out subtask, create a sample
                for prompt, emb in self.query_embeddings.items():
                    if self.task_to_subtask.get(prompt) != held_out:
                        continue

                    psi = self._compute_psi(emb, exclude_subtask=held_out)
                    label = region.utility_by_subtask[held_out]

                    for method in ["baseline", "elementwise", "bilinear"]:
                        train_X[method].append(self._build_features(region_pattern, psi, method))
                    train_y.append(label)

            # Collect test samples: predict region utility on held_out
            test_X = {"baseline": [], "elementwise": [], "bilinear": []}
            test_y = []
            test_region_ids = []

            for region in self.regions:
                if held_out not in region.utility_by_subtask:
                    continue

                region_pattern = np.array([
                    region.utility_by_subtask.get(st, 0.5)
                    for st in train_subtasks
                ])

                # Use a representative query from held_out (or average)
                held_out_embs = [
                    emb for prompt, emb in self.query_embeddings.items()
                    if self.task_to_subtask.get(prompt) == held_out
                ]
                if not held_out_embs:
                    continue

                avg_emb = np.mean(held_out_embs, axis=0)
                psi = self._compute_psi(avg_emb, exclude_subtask=held_out)
                label = region.utility_by_subtask[held_out]

                for method in ["baseline", "elementwise", "bilinear"]:
                    test_X[method].append(self._build_features(region_pattern, psi, method))
                test_y.append(label)
                test_region_ids.append(region.region_id)

            if len(test_y) < 3 or len(train_y) < 10:
                logger.warning(f"Not enough data for {held_out}: train={len(train_y)}, test={len(test_y)}")
                continue

            train_y = np.array(train_y)
            test_y = np.array(test_y)

            logger.info(f"Train: {len(train_y)} samples, Test: {len(test_y)} regions")
            logger.info(f"Train utility: {train_y.mean():.3f}±{train_y.std():.3f}")
            logger.info(f"Test utility:  {test_y.mean():.3f}±{test_y.std():.3f}")

            # Train and evaluate each method
            for method in ["baseline", "elementwise", "bilinear"]:
                X_tr = np.array(train_X[method])
                X_te = np.array(test_X[method])

                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                model.fit(X_tr, train_y)

                pred = model.predict(X_te)
                pred = np.clip(pred, 0, 1)

                metrics = self._evaluate(pred, test_y, method)
                results[method].append(metrics)

                logger.info(f"  [{method:12s}] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}  Spearman={metrics['spearman']:.4f}")

        self._print_summary(results)
        return results

    def _evaluate(self, pred: np.ndarray, true: np.ndarray, method_name: str) -> dict:
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        spearman = 0.0
        if len(pred) >= 3 and np.std(pred) > 1e-8 and np.std(true) > 1e-8:
            spearman, _ = spearmanr(pred, true)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "spearman": float(spearman),
            "n_samples": int(len(true)),
        }

    def _print_summary(self, results: dict):
        logger.info(f"\n{'='*70}")
        logger.info("PILOT ZERO-SHOT PREDICTOR SUMMARY (Reconstructed Regions)")
        logger.info(f"{'='*70}\n")
        logger.info(f"{'Method':<15s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'Spearman':>10s}")
        logger.info("-" * 60)

        for method in ["baseline", "elementwise", "bilinear"]:
            errors = results[method]
            if not errors:
                continue

            mae = np.mean([e["mae"] for e in errors])
            rmse = np.mean([e["rmse"] for e in errors])
            r2 = np.mean([e["r2"] for e in errors])
            spearman = np.mean([e["spearman"] for e in errors])

            logger.info(f"{method:<15s} {mae:8.4f} {rmse:8.4f} {r2:8.4f} {spearman:10.4f}")

        logger.info("\nInterpretation:")
        logger.info("  - R² > 0 → predictor has signal (better than mean)")
        logger.info("  - elementwise > baseline → interaction features help")
        logger.info("  - bilinear > elementwise → cross-subtask transfer matters")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot = PilotZeroShotReconstructed()
    results = pilot.run_leave_one_out()

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
