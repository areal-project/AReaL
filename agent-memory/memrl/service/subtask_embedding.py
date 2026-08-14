"""
Zero-shot region transfer via subtask embedding similarity.

Key idea:
- Each subtask has a semantic embedding (from task examples or description)
- When target_subtask is unseen, find the most similar known subtask
- Transfer utility from similar subtask to target subtask
- Weighted by similarity confidence
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SubtaskEmbeddingManager:
    """
    Manages subtask embeddings for zero-shot transfer.

    Two modes:
    1. Description-based: embed subtask descriptions
    2. Example-based: average embeddings of tasks in each subtask
    """

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.subtask_embeddings: Dict[str, np.ndarray] = {}
        self._similarity_cache: Dict[tuple, float] = {}

    def add_subtask_from_description(self, subtask_name: str, description: str):
        """Add subtask embedding from description."""
        if self.embedding_model is None:
            raise ValueError("embedding_model required for description-based mode")

        embedding = self.embedding_model.encode(description)
        self.subtask_embeddings[subtask_name] = np.array(embedding)
        logger.info(f"Added subtask embedding for {subtask_name} (description-based)")

    def add_subtask_from_examples(
        self,
        subtask_name: str,
        task_queries: List[str],
        max_examples: int = 100
    ):
        """
        Add subtask embedding from task examples.

        Args:
            subtask_name: Name of the subtask
            task_queries: List of task query strings in this subtask
            max_examples: Max number of examples to average (for efficiency)
        """
        if self.embedding_model is None:
            raise ValueError("embedding_model required for example-based mode")

        if not task_queries:
            logger.warning(f"No examples for subtask {subtask_name}")
            return

        # Sample if too many examples
        if len(task_queries) > max_examples:
            import random
            task_queries = random.sample(task_queries, max_examples)

        # Compute average embedding
        embeddings = [self.embedding_model.encode(q) for q in task_queries]
        avg_embedding = np.mean(embeddings, axis=0)

        self.subtask_embeddings[subtask_name] = avg_embedding
        logger.info(
            f"Added subtask embedding for {subtask_name} "
            f"(averaged {len(task_queries)} examples)"
        )

    def compute_similarity(self, subtask_a: str, subtask_b: str) -> float:
        """Compute cosine similarity between two subtasks."""
        cache_key = tuple(sorted([subtask_a, subtask_b]))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        emb_a = self.subtask_embeddings.get(subtask_a)
        emb_b = self.subtask_embeddings.get(subtask_b)

        if emb_a is None or emb_b is None:
            return 0.0

        # Cosine similarity
        sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        sim = float(np.clip(sim, -1.0, 1.0))

        self._similarity_cache[cache_key] = sim
        return sim

    def find_most_similar_subtask(
        self,
        target_subtask: str,
        candidate_subtasks: List[str],
        min_similarity: float = 0.3
    ) -> Optional[tuple]:
        """
        Find the most similar known subtask to target.

        Returns:
            (best_subtask, similarity) or None if no good match
        """
        if target_subtask not in self.subtask_embeddings:
            logger.warning(f"Target subtask {target_subtask} has no embedding")
            return None

        best_subtask = None
        best_sim = -1.0

        for candidate in candidate_subtasks:
            if candidate == target_subtask:
                continue

            sim = self.compute_similarity(target_subtask, candidate)
            if sim > best_sim:
                best_sim = sim
                best_subtask = candidate

        if best_sim < min_similarity:
            logger.debug(
                f"No good match for {target_subtask} "
                f"(best: {best_subtask} with sim={best_sim:.3f})"
            )
            return None

        return (best_subtask, best_sim)

    def get_transfer_weights(
        self,
        target_subtask: str,
        source_subtasks: List[str],
        temperature: float = 2.0
    ) -> Dict[str, float]:
        """
        Get weighted transfer from multiple source subtasks.

        Returns:
            {source_subtask: weight} where weights sum to 1
        """
        if target_subtask not in self.subtask_embeddings:
            return {}

        similarities = {}
        for src in source_subtasks:
            if src == target_subtask:
                continue
            sim = self.compute_similarity(target_subtask, src)
            if sim > 0:
                similarities[src] = sim

        if not similarities:
            return {}

        # Softmax with temperature
        sims = np.array(list(similarities.values()))
        exp_sims = np.exp(sims / temperature)
        weights = exp_sims / exp_sims.sum()

        return {
            src: float(w)
            for src, w in zip(similarities.keys(), weights)
        }

    def save(self, path: str):
        """Save subtask embeddings."""
        import json
        state = {
            "subtask_embeddings": {
                k: v.tolist() for k, v in self.subtask_embeddings.items()
            }
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"SubtaskEmbeddingManager saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SubtaskEmbeddingManager":
        """Load subtask embeddings."""
        import json
        with open(path, "r") as f:
            state = json.load(f)

        mgr = cls(embedding_model=None)
        mgr.subtask_embeddings = {
            k: np.array(v) for k, v in state["subtask_embeddings"].items()
        }
        logger.info(f"SubtaskEmbeddingManager loaded from {path}")
        return mgr
