"""
Pilot Experiment: Validate Advanced Transfer Mechanisms

Goal: Verify that advanced transfer mechanisms (multiple prototypes + learned calibration)
      improve zero-shot transfer accuracy before investing in full experiments.

Experimental Design:
1. Use BCB training data (7 subtasks)
2. Leave-one-subtask-out cross-validation
3. For each held-out subtask:
   - Build prototypes from remaining 6 subtasks
   - Fit calibration model on remaining 6 subtasks
   - Predict transfer weights to held-out subtask
   - Compare with ground truth utility correlations

Metrics:
- Transfer prediction error (MAE, RMSE)
- Ranking correlation (Spearman's ρ)
- Negative transfer detection rate

Baselines:
A. Raw embedding similarity (single mean prototype, no calibration)
B. Multiple prototypes only (no calibration)
C. Calibration only (single prototype)
D. Full system (multiple prototypes + calibration)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransferMechanismPilot:
    """
    Pilot experiment to validate advanced transfer mechanisms.
    """

    def __init__(
        self,
        bcb_checkpoint_path: str,
        embedding_model,
    ):
        self.bcb_checkpoint_path = Path(bcb_checkpoint_path)
        self.embedding_model = embedding_model

        # Load BCB training data
        self.subtask_utilities = {}  # {subtask: {mem_id: utility}}
        self.subtask_tasks = {}  # {subtask: [task_queries]}
        self.subtask_correlations = {}  # {(src, tgt): correlation}

        self._load_bcb_data()

    def _load_bcb_data(self):
        """
        Load BCB per-subtask utilities and task queries.
        """
        logger.info(f"Loading BCB data from {self.bcb_checkpoint_path}")

        # Load region manager to get per-subtask Q values
        region_mgr_path = self.bcb_checkpoint_path / "region_manager.json"
        if not region_mgr_path.exists():
            raise FileNotFoundError(f"Region manager not found: {region_mgr_path}")

        with open(region_mgr_path, "r") as f:
            region_state = json.load(f)

        # Extract per-subtask Q: {mem_id: {subtask: q_value}}
        subtask_q = region_state.get("subtask_q", {})

        # Reorganize to {subtask: {mem_id: utility}}
        for mem_id, q_dict in subtask_q.items():
            for subtask, utility in q_dict.items():
                if subtask not in self.subtask_utilities:
                    self.subtask_utilities[subtask] = {}
                self.subtask_utilities[subtask][mem_id] = utility

        logger.info(f"Loaded utilities for {len(self.subtask_utilities)} subtasks")

        # Load task queries from training logs
        # (This would need to be adapted based on actual log format)
        # For now, we'll use placeholder
        logger.warning("Task query loading not implemented - using placeholder")

    def compute_ground_truth_correlations(self):
        """
        Compute ground truth utility correlations between all subtask pairs.

        This is the "empirical transfer matrix" we want to predict.
        """
        subtasks = list(self.subtask_utilities.keys())

        for i, src in enumerate(subtasks):
            for tgt in subtasks[i+1:]:
                # Find common memories
                src_mems = set(self.subtask_utilities[src].keys())
                tgt_mems = set(self.subtask_utilities[tgt].keys())
                common_mems = src_mems & tgt_mems

                if len(common_mems) < 10:
                    continue

                # Compute correlation of utilities
                src_utils = [self.subtask_utilities[src][m] for m in common_mems]
                tgt_utils = [self.subtask_utilities[tgt][m] for m in common_mems]

                corr = np.corrcoef(src_utils, tgt_utils)[0, 1]
                self.subtask_correlations[(src, tgt)] = corr
                self.subtask_correlations[(tgt, src)] = corr  # Symmetric

        logger.info(
            f"Computed {len(self.subtask_correlations)} ground truth correlations"
        )

    def run_leave_one_out_validation(self):
        """
        Leave-one-subtask-out cross-validation.

        For each held-out subtask:
        1. Build prototypes from remaining subtasks
        2. Fit calibration on remaining subtasks
        3. Predict transfer to held-out subtask
        4. Compare with ground truth
        """
        from memrl.service.advanced_transfer import AdvancedTransferManager

        subtasks = list(self.subtask_utilities.keys())
        results = {
            "baseline_raw": [],  # Raw similarity, single prototype
            "baseline_multi_proto": [],  # Multiple prototypes, no calibration
            "baseline_calibration": [],  # Single prototype, with calibration
            "full_system": [],  # Multiple prototypes + calibration
        }

        for held_out in subtasks:
            logger.info(f"\n{'='*60}")
            logger.info(f"Held-out subtask: {held_out}")
            logger.info(f"{'='*60}")

            train_subtasks = [s for s in subtasks if s != held_out]

            # Ground truth: actual correlations to held-out subtask
            ground_truth = {}
            for src in train_subtasks:
                key = (src, held_out)
                if key in self.subtask_correlations:
                    ground_truth[src] = self.subtask_correlations[key]

            if not ground_truth:
                logger.warning(f"No ground truth for {held_out}, skipping")
                continue

            # === Baseline A: Raw similarity, single prototype ===
            pred_raw = self._predict_raw_similarity(train_subtasks, held_out)
            error_raw = self._compute_prediction_error(pred_raw, ground_truth)
            results["baseline_raw"].append(error_raw)
            logger.info(f"Baseline (raw similarity): MAE={error_raw['mae']:.3f}")

            # === Baseline B: Multiple prototypes, no calibration ===
            pred_multi = self._predict_multi_prototype(train_subtasks, held_out)
            error_multi = self._compute_prediction_error(pred_multi, ground_truth)
            results["baseline_multi_proto"].append(error_multi)
            logger.info(f"Baseline (multi-proto): MAE={error_multi['mae']:.3f}")

            # === Baseline C: Single prototype, with calibration ===
            pred_calib = self._predict_with_calibration(
                train_subtasks, held_out, use_multi_proto=False
            )
            error_calib = self._compute_prediction_error(pred_calib, ground_truth)
            results["baseline_calibration"].append(error_calib)
            logger.info(f"Baseline (calibration): MAE={error_calib['mae']:.3f}")

            # === Full System: Multiple prototypes + calibration ===
            pred_full = self._predict_with_calibration(
                train_subtasks, held_out, use_multi_proto=True
            )
            error_full = self._compute_prediction_error(pred_full, ground_truth)
            results["full_system"].append(error_full)
            logger.info(f"Full system: MAE={error_full['mae']:.3f}")

        # Aggregate results
        self._print_summary(results)
        return results

    def _predict_raw_similarity(
        self, train_subtasks: List[str], target_subtask: str
    ) -> Dict[str, float]:
        """Baseline: raw cosine similarity with single mean prototype."""
        # TODO: Implement using actual task embeddings
        # For now, return placeholder
        return {src: 0.5 for src in train_subtasks}

    def _predict_multi_prototype(
        self, train_subtasks: List[str], target_subtask: str
    ) -> Dict[str, float]:
        """Baseline: multiple prototypes, no calibration."""
        # TODO: Implement
        return {src: 0.5 for src in train_subtasks}

    def _predict_with_calibration(
        self,
        train_subtasks: List[str],
        target_subtask: str,
        use_multi_proto: bool,
    ) -> Dict[str, float]:
        """Full system or calibration-only baseline."""
        # TODO: Implement
        return {src: 0.5 for src in train_subtasks}

    def _compute_prediction_error(
        self,
        predictions: Dict[str, float],
        ground_truth: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute prediction error metrics.

        Returns:
            {
                "mae": mean absolute error,
                "rmse": root mean squared error,
                "spearman": Spearman's rank correlation,
            }
        """
        from scipy.stats import spearmanr

        # Align predictions and ground truth
        common_keys = set(predictions.keys()) & set(ground_truth.keys())
        if not common_keys:
            return {"mae": float("inf"), "rmse": float("inf"), "spearman": 0.0}

        pred_vals = [predictions[k] for k in common_keys]
        true_vals = [ground_truth[k] for k in common_keys]

        mae = np.mean(np.abs(np.array(pred_vals) - np.array(true_vals)))
        rmse = np.sqrt(np.mean((np.array(pred_vals) - np.array(true_vals)) ** 2))
        spearman, _ = spearmanr(pred_vals, true_vals)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "spearman": float(spearman),
        }

    def _print_summary(self, results: Dict[str, List[Dict]]):
        """Print summary of pilot results."""
        logger.info(f"\n{'='*60}")
        logger.info("PILOT EXPERIMENT SUMMARY")
        logger.info(f"{'='*60}")

        for method, errors in results.items():
            if not errors:
                continue

            mae_mean = np.mean([e["mae"] for e in errors])
            mae_std = np.std([e["mae"] for e in errors])
            rmse_mean = np.mean([e["rmse"] for e in errors])
            spearman_mean = np.mean([e["spearman"] for e in errors])

            logger.info(f"\n{method}:")
            logger.info(f"  MAE:      {mae_mean:.3f} ± {mae_std:.3f}")
            logger.info(f"  RMSE:     {rmse_mean:.3f}")
            logger.info(f"  Spearman: {spearman_mean:.3f}")


def main():
    """Run pilot experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Pilot: Validate transfer mechanisms")
    parser.add_argument(
        "--bcb_checkpoint",
        type=str,
        required=True,
        help="Path to BCB region checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pilot_transfer_mechanisms",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # TODO: Initialize embedding model
    embedding_model = None

    pilot = TransferMechanismPilot(args.bcb_checkpoint, embedding_model)
    pilot.compute_ground_truth_correlations()
    results = pilot.run_leave_one_out_validation()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
