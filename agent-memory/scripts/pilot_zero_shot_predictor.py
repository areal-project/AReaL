#!/usr/bin/env python3
"""
Pilot: Validate Zero-Shot Utility Predictor

Validates that we can predict u(region, unseen_subtask) using:
  features = [region_pattern, ψ, region_pattern ⊙ ψ]
  where ψ = [sim(query_emb, centroid_st) for st in known_subtasks]

Evaluation: Leave-one-subtask-out cross-validation on BCB data.

Methods compared:
  A. No gating (baseline)
  B. Generalist score (mean utility, no embedding info)
  C. Elementwise interaction + Ridge
  D. Bilinear + Ridge
  E. Bilinear + per-region bias

Usage:
    cd /storage/openpsi/users/yl/agent-memory/MemRL
    python scripts/pilot_zero_shot_predictor.py
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

EXP_BASE = Path(
    "results/deepseek_region_local_embed_b32/bigcodebench_eval/instruct_full/region/"
    "20260509_212923_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region"
)
PILOT_Q_PATH = Path("results/pilot_utility_consistency/reconstructed_per_subtask_q.json")
OUTPUT_DIR = Path("results/pilot_zero_shot_predictor")


def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class PilotZeroShotPredictor:

    def __init__(self):
        self.subtask_q: Dict[str, Dict[str, float]] = {}   # {mem_id: {subtask: q}}
        self.query_embeddings: Dict[str, np.ndarray] = {}   # {task_prompt: emb}
        self.task_info: Dict[str, dict] = {}                # {task_id: {subtask, reward, prompt, selected_mems}}
        self.subtask_centroids: Dict[str, np.ndarray] = {}  # {subtask: centroid_emb}
        self.known_subtasks: List[str] = []

        self._load_data()

    # ===== Data Loading =====

    def _load_data(self):
        logger.info("Loading data...")

        # 1. Per-subtask Q
        with open(PILOT_Q_PATH) as f:
            self.subtask_q = json.load(f)
        logger.info(f"  Per-subtask Q: {len(self.subtask_q)} memories")

        # 2. Query embeddings
        emb_path = EXP_BASE / "epoch10/snapshot/10/local_cache/query_embeddings.json"
        with open(emb_path) as f:
            raw = json.load(f)
        self.query_embeddings = {k: np.array(v) for k, v in raw.items()}
        logger.info(f"  Query embeddings: {len(self.query_embeddings)} entries, dim={next(iter(self.query_embeddings.values())).shape[0]}")

        # 3. Task info from all epochs (accumulate more data)
        self._load_all_epochs()

        # 4. Compute subtask centroids
        self._compute_centroids()

    def _load_all_epochs(self):
        for epoch in range(1, 11):
            samples_path = EXP_BASE / f"epoch{epoch}/train/samples.jsonl"
            retrieval_path = EXP_BASE / f"epoch{epoch}/train/memory_retrieval.jsonl"

            if not samples_path.exists() or not retrieval_path.exists():
                continue

            # Load retrieval: task_id → selected_mems
            retrieval = {}
            with open(retrieval_path) as f:
                for line in f:
                    entry = json.loads(line)
                    retrieval[entry["task_id"]] = entry.get("selected_ids", [])

            # Load samples: task_id → {subtask, reward, prompt}
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
                    self.task_info[key] = {
                        "task_id": task_id,
                        "epoch": epoch,
                        "subtask": subtask,
                        "reward": reward,
                        "prompt": prompt,
                        "selected_mems": selected_mems,
                    }

        logger.info(f"  Task info: {len(self.task_info)} (task, epoch) entries")

        subtask_counts = defaultdict(int)
        for v in self.task_info.values():
            subtask_counts[v["subtask"]] += 1
        for st in sorted(subtask_counts):
            logger.info(f"    {st}: {subtask_counts[st]}")
        self.known_subtasks = sorted(subtask_counts.keys())

    def _compute_centroids(self):
        """Compute subtask centroids from query embeddings."""
        subtask_embs = defaultdict(list)

        for info in self.task_info.values():
            prompt = info["prompt"]
            if prompt in self.query_embeddings:
                subtask_embs[info["subtask"]].append(self.query_embeddings[prompt])

        for subtask, embs in subtask_embs.items():
            self.subtask_centroids[subtask] = np.mean(embs, axis=0)
            logger.info(f"  Centroid for {subtask}: {len(embs)} queries averaged")

    # ===== Feature Construction =====

    def _compute_psi(self, query_emb: np.ndarray, exclude_subtask: str = None) -> np.ndarray:
        """ψ = [sim(query_emb, centroid_st) for st in known_subtasks]"""
        sims = []
        for st in self.known_subtasks:
            if st == exclude_subtask or st not in self.subtask_centroids:
                sims.append(0.0)
            else:
                sims.append(cosine_similarity(query_emb, self.subtask_centroids[st]))
        return np.array(sims)

    def _compute_region_pattern(
        self, mem_ids: List[str], exclude_subtask: str = None
    ) -> np.ndarray:
        """
        Region pattern = average per-subtask Q across the memories used.
        (Since we don't have explicit region IDs, we use selected memories as a proxy.)
        """
        pattern = []
        for st in self.known_subtasks:
            if st == exclude_subtask:
                pattern.append(0.0)
                continue
            q_vals = []
            for mid in mem_ids:
                if mid in self.subtask_q and st in self.subtask_q[mid]:
                    q_vals.append(self.subtask_q[mid][st])
            if q_vals:
                pattern.append(np.mean(q_vals))
            else:
                pattern.append(0.5)
        return np.array(pattern)

    def _build_features(
        self, region_pattern: np.ndarray, psi: np.ndarray, method: str
    ) -> np.ndarray:
        """Build feature vector for given method."""
        if method == "generalist":
            return region_pattern
        elif method == "elementwise":
            return np.concatenate([region_pattern, psi, region_pattern * psi])
        elif method == "bilinear":
            outer = np.outer(region_pattern, psi).flatten()
            return np.concatenate([region_pattern, psi, outer])
        else:
            raise ValueError(f"Unknown method: {method}")

    # ===== Leave-One-Subtask-Out Validation =====

    def run_leave_one_out(self):
        """Main pilot experiment."""
        results = {
            "no_gating": [],
            "generalist": [],
            "elementwise": [],
            "bilinear": [],
        }

        for held_out in self.known_subtasks:
            if held_out not in self.subtask_centroids:
                logger.warning(f"No centroid for {held_out}, skipping")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Held-out subtask: {held_out}")
            logger.info(f"{'='*60}")

            train_subtasks = [s for s in self.known_subtasks if s != held_out]

            # Collect training samples (from tasks in non-held-out subtasks)
            train_X = {"elementwise": [], "bilinear": [], "generalist": []}
            train_y = []

            # Collect test samples (from tasks in held-out subtask)
            test_X = {"elementwise": [], "bilinear": [], "generalist": []}
            test_y = []

            for key, info in self.task_info.items():
                prompt = info["prompt"]
                if prompt not in self.query_embeddings:
                    continue
                if not info["selected_mems"]:
                    continue

                query_emb = self.query_embeddings[prompt]
                psi = self._compute_psi(query_emb, exclude_subtask=held_out)
                region_pattern = self._compute_region_pattern(
                    info["selected_mems"], exclude_subtask=held_out
                )
                reward = info["reward"]

                if info["subtask"] == held_out:
                    # Test sample
                    for method in ["elementwise", "bilinear", "generalist"]:
                        test_X[method].append(self._build_features(region_pattern, psi, method))
                    test_y.append(reward)
                else:
                    # Training sample
                    for method in ["elementwise", "bilinear", "generalist"]:
                        train_X[method].append(self._build_features(region_pattern, psi, method))
                    train_y.append(reward)

            if len(test_y) < 5 or len(train_y) < 20:
                logger.warning(f"Not enough data for {held_out}: train={len(train_y)}, test={len(test_y)}")
                continue

            train_y = np.array(train_y)
            test_y = np.array(test_y)

            logger.info(f"Train: {len(train_y)} samples, Test: {len(test_y)} samples")
            logger.info(f"Train pass rate: {train_y.mean():.3f}, Test pass rate: {test_y.mean():.3f}")

            # === Method A: No gating (predict global mean) ===
            pred_no_gating = np.full(len(test_y), train_y.mean())
            metrics_no_gating = self._evaluate(pred_no_gating, test_y, "No gating")
            results["no_gating"].append(metrics_no_gating)

            # === Method B-D: Supervised predictors ===
            for method in ["generalist", "elementwise", "bilinear"]:
                X_tr = np.array(train_X[method])
                X_te = np.array(test_X[method])

                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
                model.fit(X_tr, train_y)

                pred = model.predict(X_te)
                pred = np.clip(pred, 0, 1)

                metrics = self._evaluate(pred, test_y, method)
                results[method].append(metrics)

                logger.info(f"  [{method:15s}] MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R²={metrics['r2']:.4f}")

        self._print_summary(results)
        return results

    def _evaluate(self, pred: np.ndarray, true: np.ndarray, method_name: str) -> dict:
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))

        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        spearman = 0.0
        if len(pred) >= 3 and np.std(pred) > 1e-8:
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
        logger.info("PILOT ZERO-SHOT PREDICTOR SUMMARY")
        logger.info(f"{'='*70}\n")
        logger.info(f"{'Method':<20s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s} {'Spearman':>10s}")
        logger.info("-" * 60)

        for method in ["no_gating", "generalist", "elementwise", "bilinear"]:
            errors = results[method]
            if not errors:
                continue

            mae = np.mean([e["mae"] for e in errors])
            rmse = np.mean([e["rmse"] for e in errors])
            r2 = np.mean([e["r2"] for e in errors])
            spearman = np.mean([e["spearman"] for e in errors])

            logger.info(f"{method:<20s} {mae:8.4f} {rmse:8.4f} {r2:8.4f} {spearman:10.4f}")

        logger.info("")
        logger.info("Interpretation:")
        logger.info("  - Lower MAE/RMSE = better prediction accuracy")
        logger.info("  - Higher R²/Spearman = better at distinguishing useful vs useless regions")
        logger.info("  - elementwise > no_gating → interaction features help zero-shot transfer")
        logger.info("  - bilinear > elementwise → cross-subtask transfer patterns matter")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pilot = PilotZeroShotPredictor()
    results = pilot.run_leave_one_out()

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
