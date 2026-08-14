"""
Task hierarchy definitions for region-based memory transfer.

Defines subtask structure for each benchmark, used by RegionManager
to track per-subtask utility statistics.

Supports two modes for BCB:
- Domain-level subtasks (7 human-defined domains) — legacy
- Auto task clusters (K data-driven clusters from task embeddings) — new default
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class TaskClusterManager:
    """Data-driven task clustering using task embeddings.

    Clusters BCB tasks into K groups via KMeans on cosine-normalized
    embeddings. Each cluster becomes a subtask for Q tracking.
    """

    def __init__(self, K: int = 12):
        self.K = K
        self.centroids: Optional[np.ndarray] = None  # [K, d]
        self._fitted = False

    def fit(self, task_embeddings: Dict[str, np.ndarray]) -> None:
        """Fit KMeans on task embeddings."""
        if not task_embeddings:
            logger.warning("No task embeddings for clustering")
            return

        ids = list(task_embeddings.keys())
        X = np.array([task_embeddings[tid] for tid in ids])

        # Cosine normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_norm = X / norms

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(self.K, len(ids)), random_state=42, n_init=10)
        km.fit(X_norm)

        self.centroids = km.cluster_centers_  # already normalized from normalized input
        self._fitted = True

        # Log cluster sizes
        from collections import Counter
        sizes = Counter(km.labels_)
        logger.info(
            "TaskClusterManager fitted: K=%d, %d tasks, cluster sizes: %s",
            self.K, len(ids), dict(sorted(sizes.items())),
        )

    def save(self, path: str) -> None:
        """Save centroids to file for checkpoint resume."""
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "K": self.K,
            "centroids": self.centroids.tolist() if self.centroids is not None else None,
            "fitted": self._fitted,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str) -> bool:
        """Load centroids from file. Returns True if successful."""
        import json
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("centroids") and data.get("fitted"):
                self.centroids = np.array(data["centroids"])
                self.K = data["K"]
                self._fitted = True
                logger.info("TaskClusterManager loaded: K=%d from %s", self.K, path)
                return True
        except Exception as e:
            logger.warning("Failed to load task clusters from %s: %s", path, e)
        return False

    def predict(self, embedding: np.ndarray) -> int:
        """Find nearest cluster for a task embedding. Returns -1 if not fitted or invalid."""
        if not self._fitted or self.centroids is None:
            return -1

        emb = np.array(embedding).flatten()
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            return -1
        emb_norm = emb / norm

        sims = self.centroids @ emb_norm
        return int(np.argmax(sims))

    def get_subtask(self, embedding: np.ndarray, prefix: str = "bcb") -> str:
        """Return subtask string for a task embedding."""
        cluster_id = self.predict(embedding)
        if cluster_id < 0:
            return f"{prefix}/unknown"
        return f"{prefix}/cluster_{cluster_id}"


# Global instance (initialized lazily by RegionMemoryService or runner)
_task_cluster_mgr: Optional[TaskClusterManager] = None


def get_task_cluster_manager(K: int = 12) -> TaskClusterManager:
    """Get or create the global TaskClusterManager."""
    global _task_cluster_mgr
    if _task_cluster_mgr is None:
        _task_cluster_mgr = TaskClusterManager(K=K)
    return _task_cluster_mgr


def get_task_hierarchy(K: int = 12) -> Dict[str, Dict[str, Any]]:
    """Build task hierarchy with dynamic K for BCB clusters."""
    return {
        "bigcodebench": {
            "subtasks": [f"bcb/cluster_{i}" for i in range(K)] + ["bcb/unknown"],
            "subtask_field": "auto_cluster",
        },
        "alfworld": {
            "subtasks": [
                "alf/look_at_obj_in_light",
                "alf/pick_and_place_simple",
                "alf/pick_and_place_with_movable_recep",
                "alf/pick_clean_then_place_in_recep",
                "alf/pick_cool_then_place_in_recep",
                "alf/pick_heat_then_place_in_recep",
                "alf/pick_two_obj_and_place",
                "alf/unknown",
            ],
            "subtask_field": "task_type",
        },
        "hle": {
            "subtasks": [
                "hle/math", "hle/physics", "hle/chemistry",
                "hle/biology_medicine", "hle/computer_science_ai", "hle/engineering",
                "hle/humanities_social_science", "hle/other",
            ],
            "subtask_field": "category",
        },
        "llb_os": {
            "subtasks": [
                "llb_os/text_processing",
                "llb_os/user_group",
                "llb_os/user_admin",
                "llb_os/file_ops",
                "llb_os/config_setup",
                "llb_os/unknown",
            ],
            "subtask_field": "skill_category",
        },
        "llb_db": {
            "subtasks": [
                "llb_db/insert",
                "llb_db/update",
                "llb_db/delete",
                "llb_db/select_subquery",
                "llb_db/select_aggregate",
                "llb_db/select_simple",
                "llb_db/unknown",
            ],
            "subtask_field": "skill_category",
        },
    }


# Default instance (backward compat)
TASK_HIERARCHY = get_task_hierarchy(12)


def get_db_multi_axis_subtasks(skill_list: List[str]) -> List[Tuple[str, float]]:
    """Return normalized multi-axis DB labels for Q/evidence supervision.

    Axis mass is fixed so tasks with many skills do not contribute more total
    evidence: operation=0.30, query-shape=0.30, modifiers(total)=0.25,
    complexity=0.15. Active modifiers divide their axis mass equally.
    """
    skills = set(skill_list or [])
    labels: List[Tuple[str, float]] = []

    # Operation axis.
    if "insert" in skills:
        operation = "insert"
    elif "update" in skills:
        operation = "update"
    elif "delete" in skills:
        operation = "delete"
    else:
        operation = "select"
    labels.append((f"llb_db/op/{operation}", 0.30))

    # Query-shape axis.
    if operation != "select":
        shape = "mutation"
    elif skills & {"subquery_nested", "subquery_multiple", "subquery_single"}:
        shape = "subquery"
    elif skills & {
        "group_by_single_column", "group_by_multiple_columns",
        "having_single_condition_with_aggregate",
        "having_multiple_conditions_with_aggregate",
        "having_aggregate_calculation",
    }:
        shape = "aggregate"
    else:
        shape = "simple"
    labels.append((f"llb_db/shape/{shape}", 0.30))

    # Modifier axis. Multiple active modifiers share a fixed total mass.
    modifier_groups = {
        "predicate": {"where_single_condition", "where_multiple_conditions", "where_nested_conditions"},
        "group_having": {
            "group_by_single_column", "group_by_multiple_columns",
            "having_single_condition_with_aggregate",
            "having_multiple_conditions_with_aggregate", "having_aggregate_calculation",
        },
        "ordering": {
            "order_by_single_column", "order_by_multiple_columns_same_direction",
            "order_by_multiple_columns_different_directions",
        },
        "pagination": {"limit_only", "limit_and_offset"},
        "aliasing": {"column_alias", "table_alias"},
        "nested": {"subquery_nested", "subquery_multiple", "subquery_single"},
    }
    active = [name for name, members in modifier_groups.items() if skills & members]
    if not active:
        active = ["none"]
    modifier_weight = 0.25 / len(active)
    labels.extend((f"llb_db/mod/{name}", modifier_weight) for name in active)

    # Complexity axis based on composition width.
    n_skills = len(skills)
    complexity = "low" if n_skills <= 2 else ("medium" if n_skills <= 5 else "high")
    labels.append((f"llb_db/complexity/{complexity}", 0.15))

    total = sum(weight for _, weight in labels)
    if total <= 0:
        return [("llb_db/unknown", 1.0)]
    return [(label, weight / total) for label, weight in labels]


def get_primary_subtask(benchmark: str, metadata: Dict[str, Any]) -> str:
    """Extract the primary subtask type from task metadata.

    For BCB: uses auto task cluster if embedding available, else domain fallback.
    For HLE: uses category.
    """
    if benchmark == "bigcodebench":
        # Auto cluster mode: use task embedding
        embedding = metadata.get("embedding")
        if embedding is not None and _task_cluster_mgr is not None and _task_cluster_mgr._fitted:
            return _task_cluster_mgr.get_subtask(embedding)

        # Fallback to domain-level
        domains = metadata.get("domains", [])
        if domains and isinstance(domains, list):
            return f"bcb/{domains[0]}"
        return "bcb/unknown"

    elif benchmark in ("alfworld", "alf"):
        task_type = metadata.get("task_type", "")
        if task_type:
            task_type = task_type.split("/")[0]
            # Match against known task types (handles "pick_heat_then_place_in_recep-Egg-..." format)
            _ALF_TASK_TYPES = [
                "pick_and_place_with_movable_recep",
                "pick_clean_then_place_in_recep",
                "pick_cool_then_place_in_recep",
                "pick_heat_then_place_in_recep",
                "pick_two_obj_and_place",
                "pick_and_place_simple",
                "look_at_obj_in_light",
            ]
            for tt in _ALF_TASK_TYPES:
                if task_type.startswith(tt):
                    return f"alf/{tt}"
            return f"alf/{task_type}"
        # Extract from game file name (e.g., "pick_heat_then_place_in_recep-Egg-...")
        game_file = metadata.get("game_file", "")
        if game_file:
            tt = game_file.split("-")[0] if "-" in game_file else ""
            if tt:
                return f"alf/{tt}"
        return "alf/unknown"

    elif benchmark in ("hle", "HLE"):
        category = metadata.get("category", "other") or "other"
        return hle_category_to_subtask(category)

    elif benchmark == "llb_os":
        skills = set(metadata.get("skill_list", []))
        if skills & {"chsh", "chage", "gpasswd"}:
            return "llb_os/user_admin"
        elif skills & {"useradd", "usermod", "addgroup", "groupadd"}:
            return "llb_os/user_group"
        elif skills & {"grep", "awk", "wc", "sed"}:
            return "llb_os/text_processing"
        elif skills & {"find", "cp", "mv", "ln"}:
            return "llb_os/file_ops"
        else:
            return "llb_os/config_setup"

    elif benchmark == "llb_db":
        skills = set(metadata.get("skill_list", []))
        if "insert" in skills:
            return "llb_db/insert"
        if "update" in skills:
            return "llb_db/update"
        if "delete" in skills:
            return "llb_db/delete"
        if skills & {"subquery_nested", "subquery_multiple", "subquery_single"}:
            return "llb_db/select_subquery"
        if skills & {"group_by_single_column", "group_by_multiple_columns",
                     "having_single_condition_with_aggregate",
                     "having_multiple_conditions_with_aggregate",
                     "having_aggregate_calculation"}:
            return "llb_db/select_aggregate"
        if "select" in skills:
            return "llb_db/select_simple"
        return "llb_db/unknown"

    return f"{benchmark}/unknown"


# Canonical HLE categories (exactly the 8 values present in data/hle/hle_test.parquet).
# Subtask names are derived deterministically from the raw category via
# hle_category_to_subtask so the retrieval side (get_primary_subtask) and the
# registration side (_register_subtask_embeddings) can never drift apart and
# collapse everything into one subtask (which disables region clustering).
HLE_CATEGORIES = [
    "Math",
    "Physics",
    "Chemistry",
    "Biology/Medicine",
    "Computer Science/AI",
    "Engineering",
    "Humanities/Social Science",
    "Other",
]

# Short descriptions used to embed each subtask (for region transfer similarity).
HLE_SUBTASK_DESCRIPTIONS = {
    "Math": "Advanced mathematics problems including algebra, calculus, number theory, topology, and abstract algebra",
    "Physics": "Physics problems involving quantum mechanics, relativity, thermodynamics, and particle physics",
    "Chemistry": "Chemistry problems including organic synthesis, reaction mechanisms, and molecular properties",
    "Biology/Medicine": "Biology and medicine problems covering genetics, molecular biology, ecology, neuroscience, and clinical medicine",
    "Computer Science/AI": "Computer science and AI problems in algorithms, complexity theory, machine learning, and systems",
    "Engineering": "Engineering problems spanning electrical, mechanical, civil, and chemical engineering",
    "Humanities/Social Science": "Humanities and social science problems in history, sociology, economics, law, and political science",
    "Other": "Interdisciplinary or uncategorized expert-level problems",
}


def hle_category_to_subtask(category: str) -> str:
    """Deterministically slugify a raw HLE category into an ``hle/<slug>`` subtask.

    e.g. "Computer Science/AI" -> "hle/computer_science_ai",
         "Humanities/Social Science" -> "hle/humanities_social_science".
    Distinct categories always map to distinct subtasks (no merging)."""
    import re
    slug = re.sub(r"[^a-z0-9]+", "_", str(category or "other").strip().lower()).strip("_")
    return f"hle/{slug or 'other'}"


def get_benchmark_from_subtask(subtask: str) -> str:
    """Extract benchmark name from subtask string."""
    parts = subtask.split("/", 1)
    return parts[0] if parts else "unknown"


# Intra-transfer experiment splits for BCB
BCB_INTRA_SPLITS = {
    "split_1": {
        "source": ["bcb/Computation", "bcb/System", "bcb/General"],
        "target": ["bcb/Network", "bcb/Cryptography", "bcb/Time", "bcb/Visualization"],
    },
}

# Intra-transfer experiment splits for HLE
HLE_INTRA_SPLITS = {
    "split_1": {
        "source": ["hle/math", "hle/physics", "hle/computer_science_ai", "hle/engineering"],
        "target": ["hle/chemistry", "hle/biology_medicine", "hle/humanities_social_science", "hle/other"],
    },
}
