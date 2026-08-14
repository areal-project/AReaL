"""
Advanced zero-shot transfer mechanisms for region-based memory retrieval.

Implements:
1. Multiple prototypes per subtask (captures multimodal distributions)
2. Learned calibration from empirical transfer matrix
3. Negative transfer detection and mitigation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


class AdvancedTransferManager:
    """
    Manages advanced zero-shot transfer mechanisms.

    Key features:
    - Multiple prototypes per subtask (k-means clustering)
    - Learned calibration: embedding_similarity → actual_transfer_rate
    - Negative transfer detection and mitigation
    """

    def __init__(
        self,
        n_prototypes: int = 3,
        calibration_method: str = "ridge",  # "ridge" or "isotonic"
        negative_transfer_threshold: float = 2.0,
    ):
        self.n_prototypes = n_prototypes
        self.calibration_method = calibration_method
        self.negative_transfer_threshold = negative_transfer_threshold

        # Multiple prototypes: {subtask: [proto1, proto2, ...]}
        self.subtask_prototypes: Dict[str, List[np.ndarray]] = {}

        # Calibration model: embedding_sim → transfer_rate
        self.calibrator = None
        self.calibration_fitted = False

        # Empirical transfer matrix for calibration
        self.empirical_transfer_matrix: Dict[Tuple[str, str], float] = {}

        # Negative transfer risk scores
        self.negative_transfer_risks: Dict[Tuple[str, str], float] = {}

    # ========== Multiple Prototypes ==========

    def build_prototypes_from_embeddings(
        self,
        subtask_name: str,
        task_embeddings: List[np.ndarray],
        min_samples_per_cluster: int = 10,
    ):
        """
        Build multiple prototypes for a subtask using k-means.

        Args:
            subtask_name: Name of the subtask
            task_embeddings: List of task embeddings in this subtask
            min_samples_per_cluster: Minimum samples required per cluster
        """
        embeddings = np.array(task_embeddings)
        n_samples = len(embeddings)

        # Adjust n_prototypes based on sample size
        effective_n_prototypes = min(
            self.n_prototypes,
            max(1, n_samples // min_samples_per_cluster)
        )

        if effective_n_prototypes == 1:
            # Not enough samples for clustering, use mean
            prototypes = [np.mean(embeddings, axis=0)]
            logger.info(
                f"Subtask {subtask_name}: using single prototype "
                f"({n_samples} samples)"
            )
        else:
            # K-means clustering
            kmeans = KMeans(
                n_clusters=effective_n_prototypes,
                random_state=42,
                n_init=10
            )
            kmeans.fit(embeddings)
            prototypes = list(kmeans.cluster_centers_)

            # Log cluster sizes
            unique, counts = np.unique(kmeans.labels_, return_counts=True)
            cluster_info = dict(zip(unique, counts))
            logger.info(
                f"Subtask {subtask_name}: {effective_n_prototypes} prototypes, "
                f"cluster sizes: {cluster_info}"
            )

        self.subtask_prototypes[subtask_name] = prototypes

    def compute_max_prototype_similarity(
        self,
        subtask_a: str,
        subtask_b: str,
    ) -> float:
        """
        Compute maximum similarity between any prototype pair.

        This captures the best-case alignment between subtasks,
        accounting for multimodal distributions.
        """
        protos_a = self.subtask_prototypes.get(subtask_a)
        protos_b = self.subtask_prototypes.get(subtask_b)

        if protos_a is None or protos_b is None:
            return 0.0

        max_sim = -1.0
        for proto_a in protos_a:
            for proto_b in protos_b:
                sim = self._cosine_similarity(proto_a, proto_b)
                max_sim = max(max_sim, sim)

        return float(max_sim)

    def compute_avg_prototype_similarity(
        self,
        subtask_a: str,
        subtask_b: str,
    ) -> float:
        """
        Compute average similarity across all prototype pairs.

        More conservative than max similarity.
        """
        protos_a = self.subtask_prototypes.get(subtask_a)
        protos_b = self.subtask_prototypes.get(subtask_b)

        if protos_a is None or protos_b is None:
            return 0.0

        sims = []
        for proto_a in protos_a:
            for proto_b in protos_b:
                sim = self._cosine_similarity(proto_a, proto_b)
                sims.append(sim)

        return float(np.mean(sims))

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ========== Learned Calibration ==========

    def add_empirical_transfer_observation(
        self,
        source_subtask: str,
        target_subtask: str,
        transfer_rate: float,
    ):
        """
        Add an empirical observation of transfer rate.

        Args:
            source_subtask: Source subtask name
            target_subtask: Target subtask name
            transfer_rate: Observed transfer rate (e.g., correlation of utilities)
        """
        self.empirical_transfer_matrix[(source_subtask, target_subtask)] = transfer_rate

    def fit_calibration_model(self):
        """
        Fit calibration model: embedding_similarity → actual_transfer_rate.

        Uses empirical transfer matrix as ground truth.
        """
        if not self.empirical_transfer_matrix:
            logger.warning("No empirical transfer data for calibration")
            return

        # Collect (embedding_sim, empirical_transfer) pairs
        X = []  # embedding similarities
        y = []  # empirical transfer rates

        for (src, tgt), empirical_transfer in self.empirical_transfer_matrix.items():
            # Compute embedding similarity
            if self.subtask_prototypes:
                # Use max prototype similarity
                emb_sim = self.compute_max_prototype_similarity(src, tgt)
            else:
                logger.warning("No prototypes available, skipping calibration")
                return

            X.append(emb_sim)
            y.append(empirical_transfer)

        X = np.array(X).reshape(-1, 1)
        y = np.array(y)

        # Fit calibration model
        if self.calibration_method == "ridge":
            self.calibrator = Ridge(alpha=0.1)
            self.calibrator.fit(X, y)
            logger.info(
                f"Calibration fitted (Ridge): {len(X)} pairs, "
                f"R² = {self.calibrator.score(X, y):.3f}"
            )
        elif self.calibration_method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(X.flatten(), y)
            logger.info(f"Calibration fitted (Isotonic): {len(X)} pairs")
        else:
            raise ValueError(f"Unknown calibration method: {self.calibration_method}")

        self.calibration_fitted = True

    def get_calibrated_transfer_weight(
        self,
        source_subtask: str,
        target_subtask: str,
    ) -> float:
        """
        Get calibrated transfer weight from source to target.

        Returns:
            Calibrated transfer weight (can be negative for negative transfer)
        """
        # Compute embedding similarity
        emb_sim = self.compute_max_prototype_similarity(source_subtask, target_subtask)

        if not self.calibration_fitted or self.calibrator is None:
            # No calibration, return raw similarity
            return emb_sim

        # Apply calibration
        if self.calibration_method == "ridge":
            calibrated = self.calibrator.predict([[emb_sim]])[0]
        else:  # isotonic
            calibrated = self.calibrator.predict([emb_sim])[0]

        return float(calibrated)

    # ========== Negative Transfer Detection ==========

    def detect_negative_transfer_risk(
        self,
        region_utilities: Dict[str, float],
        source_subtask: str,
    ) -> float:
        """
        Detect negative transfer risk for a region.

        High risk indicators:
        - High variance in utilities (specialist region)
        - Source utility is outlier (very high or very low)

        Returns:
            Risk score (0 = safe, higher = risky)
        """
        if not region_utilities or source_subtask not in region_utilities:
            return 0.0

        utilities = list(region_utilities.values())
        source_utility = region_utilities[source_subtask]

        mean_u = np.mean(utilities)
        std_u = np.std(utilities)

        if std_u == 0:
            return 0.0

        # 1. Variance-based risk (high variance = specialist)
        variance_risk = std_u

        # 2. Outlier risk (source is far from mean)
        z_score = abs(source_utility - mean_u) / std_u
        outlier_risk = max(0, z_score - 1.0)  # Risk if z > 1

        # Combined risk
        risk = variance_risk * (1.0 + outlier_risk)

        return float(risk)

    def apply_negative_transfer_guard(
        self,
        transfer_weights: Dict[str, float],
        region_utilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply negative transfer mitigation to transfer weights.

        Args:
            transfer_weights: {source_subtask: weight}
            region_utilities: {subtask: utility} for the region

        Returns:
            Adjusted weights with negative transfer mitigation
        """
        adjusted_weights = {}

        for src, weight in transfer_weights.items():
            risk = self.detect_negative_transfer_risk(region_utilities, src)

            # Exponential penalty for high risk
            penalty = np.exp(-risk / self.negative_transfer_threshold)
            adjusted_weights[src] = weight * penalty

        # Renormalize
        total = sum(adjusted_weights.values())
        if total > 0:
            adjusted_weights = {k: v / total for k, v in adjusted_weights.items()}

        return adjusted_weights

    # ========== Persistence ==========

    def save(self, path: str):
        """Save advanced transfer manager state."""
        import json
        from pathlib import Path

        state = {
            "n_prototypes": self.n_prototypes,
            "calibration_method": self.calibration_method,
            "negative_transfer_threshold": self.negative_transfer_threshold,
            "subtask_prototypes": {
                k: [v.tolist() for v in vs]
                for k, vs in self.subtask_prototypes.items()
            },
            "empirical_transfer_matrix": {
                f"{src}→{tgt}": rate
                for (src, tgt), rate in self.empirical_transfer_matrix.items()
            },
            "calibration_fitted": self.calibration_fitted,
        }

        # Save calibrator separately if fitted
        if self.calibration_fitted and self.calibrator is not None:
            import pickle
            calibrator_path = Path(path).with_suffix(".calibrator.pkl")
            with open(calibrator_path, "wb") as f:
                pickle.dump(self.calibrator, f)
            state["calibrator_path"] = str(calibrator_path)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"AdvancedTransferManager saved to {path}")

    @classmethod
    def load(cls, path: str) -> "AdvancedTransferManager":
        """Load advanced transfer manager state."""
        import json
        import pickle
        from pathlib import Path

        with open(path, "r") as f:
            state = json.load(f)

        mgr = cls(
            n_prototypes=state["n_prototypes"],
            calibration_method=state["calibration_method"],
            negative_transfer_threshold=state["negative_transfer_threshold"],
        )

        mgr.subtask_prototypes = {
            k: [np.array(v) for v in vs]
            for k, vs in state["subtask_prototypes"].items()
        }

        mgr.empirical_transfer_matrix = {
            tuple(k.split("→")): v
            for k, v in state["empirical_transfer_matrix"].items()
        }

        mgr.calibration_fitted = state["calibration_fitted"]

        # Load calibrator if exists
        if "calibrator_path" in state:
            calibrator_path = Path(state["calibrator_path"])
            if calibrator_path.exists():
                with open(calibrator_path, "rb") as f:
                    mgr.calibrator = pickle.load(f)

        logger.info(f"AdvancedTransferManager loaded from {path}")
        return mgr
